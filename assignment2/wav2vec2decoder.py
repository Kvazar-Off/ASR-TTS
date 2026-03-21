import math
from typing import List, Tuple
from collections import defaultdict
import heapq

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        # <YOUR CODE GOES HERE>
        log_probs = torch.log_softmax(logits, dim=-1) 
        indices = torch.argmax(log_probs, dim=-1).tolist()  # (T,)

        decoded_indices = []
        for i in range(len(indices)):
            if indices[i] != self.blank_token_id and (i == 0 or indices[i] != indices[i-1]):
                decoded_indices.append(indices[i])

        return self._ids_to_text(decoded_indices)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)  # (T, V)
        T, V = log_probs.shape

        # beams[prefix] = (p_blank, p_nonblank)
        beams = {(): (0.0, float("-inf"))}

        for t in range(T):
            next_beams = defaultdict(lambda: (float("-inf"), float("-inf")))

            for prefix, (p_b, p_nb) in beams.items():
                total = _log_add(p_b, p_nb)

                # Extend with blank
                p_blank = log_probs[t, self.blank_token_id].item()
                nb_b, nb_nb = next_beams[prefix]
                next_beams[prefix] = (_log_add(nb_b, total + p_blank), nb_nb)

                # Extend with non-blank tokens
                for c in range(V):
                    if c == self.blank_token_id:
                        continue

                    p = log_probs[t, c].item()
                    last = prefix[-1] if prefix else None

                    if c == last:
                        # Repeat without blank: stay in same prefix, only from p_nb
                        nb_b, nb_nb = next_beams[prefix]
                        next_beams[prefix] = (nb_b, _log_add(nb_nb, p_nb + p))

                        # Repeat after blank: extend prefix, only from p_b
                        new_prefix = prefix + (c,)
                        nb_b, nb_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (nb_b, _log_add(nb_nb, p_b + p))
                    else:
                        # Different token: extend prefix from both p_b and p_nb
                        new_prefix = prefix + (c,)
                        nb_b, nb_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (nb_b, _log_add(nb_nb, total + p))

            # Keep only top beam_width prefixes
            top_beams = heapq.nlargest(
                self.beam_width,
                next_beams.items(),
                key=lambda item: _log_add(item[1][0], item[1][1])
            )
            beams = dict(top_beams)

        final_beams = [
            (list(prefix), _log_add(p_b, p_nb))
            for prefix, (p_b, p_nb) in beams.items()
        ]
        final_beams.sort(key=lambda x: x[1], reverse=True)

        if return_beams:
            return final_beams

        best_ids = final_beams[0][0] if final_beams else []
        return self._ids_to_text(best_ids)

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape

        wd_id = next(i for i, c in self.vocab.items() if c == self.word_delimiter)

        init_state = kenlm.State()
        self.lm_model.BeginSentenceWrite(init_state)

        # acoustic: prefix -> (p_blank, p_nonblank)
        # lm_info:  prefix -> (kenlm_state, lm_score, num_words, partial_word)
        acoustic = {(): (0.0, float("-inf"))}
        lm_info = {(): (init_state, 0.0, 0, "")}

        def extend_lm(parent_prefix, token_id):
            """Compute LM info when extending parent_prefix with token_id."""
            st, sc, nw, partial = lm_info[parent_prefix]
            if token_id == wd_id:
                if partial:
                    out_state = kenlm.State()
                    word_score = self.lm_model.BaseScore(st, partial, out_state)
                    return (out_state, sc + word_score, nw + 1, "")
                return (st, sc, nw, "")
            return (st, sc, nw, partial + self.vocab[token_id])

        def total_score(prefix, p_b, p_nb, final=False):
            ac = _log_add(p_b, p_nb)
            st, sc, nw, partial = lm_info[prefix]
            if final and partial:
                out_state = kenlm.State()
                sc = sc + self.lm_model.BaseScore(st, partial, out_state)
                nw = nw + 1
            return ac + self.alpha * sc + self.beta * nw

        for t in range(T):
            next_acoustic = defaultdict(lambda: (float("-inf"), float("-inf")))
            next_lm = {}
            log_p = log_probs[t]

            for prefix in acoustic:
                p_b, p_nb = acoustic[prefix]
                p_total = _log_add(p_b, p_nb)

                # Extend with blank — prefix stays the same
                nb_b, nb_nb = next_acoustic[prefix]
                next_acoustic[prefix] = (_log_add(nb_b, p_total + log_p[self.blank_token_id].item()), nb_nb)
                if prefix not in next_lm:
                    next_lm[prefix] = lm_info[prefix]

                for v in range(V):
                    if v == self.blank_token_id:
                        continue

                    p_v = log_p[v].item()

                    if prefix and v == prefix[-1]:
                        # Repeat without blank: stay in same prefix
                        nb_b, nb_nb = next_acoustic[prefix]
                        next_acoustic[prefix] = (nb_b, _log_add(nb_nb, p_nb + p_v))

                        # Repeat after blank: extend prefix
                        new_prefix = prefix + (v,)
                        nb_b, nb_nb = next_acoustic[new_prefix]
                        next_acoustic[new_prefix] = (nb_b, _log_add(nb_nb, p_b + p_v))
                        if new_prefix not in next_lm:
                            next_lm[new_prefix] = extend_lm(prefix, v)
                    else:
                        new_prefix = prefix + (v,)
                        nb_b, nb_nb = next_acoustic[new_prefix]
                        next_acoustic[new_prefix] = (nb_b, _log_add(nb_nb, p_total + p_v))
                        if new_prefix not in next_lm:
                            next_lm[new_prefix] = extend_lm(prefix, v)

            # Update lm_info before pruning so total_score can access it
            acoustic = dict(next_acoustic)
            lm_info = next_lm

            top_prefixes = heapq.nlargest(
                self.beam_width,
                acoustic.keys(),
                key=lambda pfx: total_score(pfx, *acoustic[pfx])
            )
            acoustic = {p: acoustic[p] for p in top_prefixes}
            lm_info = {p: lm_info[p] for p in top_prefixes}

        # Final best hypothesis — score the last partial word
        best_prefix = max(
            acoustic.keys(),
            key=lambda pfx: total_score(pfx, *acoustic[pfx], final=True)
        )
        return self._ids_to_text(list(best_prefix))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
            
        best_text = ""
        best_score = float('-inf')
        
        for token_ids, acoustic_log_prob in beams:
            text = self._ids_to_text(token_ids)
            lm_score = self.lm_model.score(text) if text else 0.0
            num_words = len(text.split()) if text else 0
            
            total_score = acoustic_log_prob + self.alpha * lm_score + self.beta * num_words
            
            if total_score > best_score:
                best_score = total_score
                best_text = text
                
        return best_text
    
    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=None)  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
