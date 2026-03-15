"""Processor class for MERaLiON2."""

from typing import List, Optional, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput


# copied from transformers.models.qwen2_audio.processing_qwen2_audio.Qwen2AudioProcessor
class MERaLiON2Processor(ProcessorMixin):
    r"""
    Constructs a MERaLiON2 processor which wraps a whisper feature extractor and a gemma tokenizer into a single processor.

    [`MERaLiON2Processor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`GemmaTokenizer`]. See the
    [`~MERaLiON2Processor.__call__`] and [`~MERaLiON2Processor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`GemmaTokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = [
        "fixed_speech_embeds_length", 
        "speech_token_index", 
        "time_duration_limit", 
        "whisper_chunk_size",
        "do_normalize"
    ]

    def __init__(
        self, 
        feature_extractor=None, 
        tokenizer=None, 
        fixed_speech_embeds_length=100,
        speech_token_index=255999,
        time_duration_limit=300,
        whisper_chunk_size=30,
        do_normalize=True
    ):
        self.fixed_speech_embeds_length = fixed_speech_embeds_length
        self.speech_token_index = speech_token_index
        self.time_duration_limit = time_duration_limit
        self.whisper_chunk_size = whisper_chunk_size
        self.number_chunk_limit = self.time_duration_limit // self.whisper_chunk_size
        self.do_normalize = do_normalize

        super().__init__(feature_extractor, tokenizer)

        self.speech_token = self.tokenizer.added_tokens_decoder[self.speech_token_index].content
        self.feature_chunk_size = self.whisper_chunk_size * self.feature_extractor.sampling_rate

    def _process_text(self, text: List[str], audio_number_chunks: np.ndarray):
        pieces = []
        for i, item in enumerate(text):
            target_string = self.speech_token * self.fixed_speech_embeds_length * audio_number_chunks[i]
            pieces.append(item.replace(self.speech_token, target_string))
        return pieces

    def _get_number_chunks(self, audios: List[np.ndarray]):
        audio_lengths = np.array([_.shape[0] for _ in audios])
        number_chunks = ((audio_lengths - 1) // self.feature_chunk_size) + 1
        return np.clip(number_chunks, a_min=None, a_max=self.number_chunk_limit)

    def _get_chunked_audios(self, audios: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(audios, np.ndarray):
            audios = [audios]

        audio_number_chunks = self._get_number_chunks(audios)
        chunked_audios = []

        for audio_idx, audio in enumerate(audios):
            for cid in range(audio_number_chunks[audio_idx]):
                chunked_audios.append(
                    audio[cid * self.feature_chunk_size: (cid + 1) * self.feature_chunk_size]
                )
        return audio_number_chunks, chunked_audios

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to GemmaTokenizer's [`~GemmaTokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
            do_normalize (`bool`, defaults to `True`):
                Whether or not to zero-mean unit-variance normalize the input. 
                Normalizing can help to significantly improve the performance of the model.
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        if not isinstance(text, list):
            text = [text]
        if not isinstance(audios, list):
            audios = [audios]
        if sampling_rate is None:
            sampling_rate = self.feature_extractor.sampling_rate
        if do_normalize is None:
            do_normalize = self.do_normalize

        for i, audio in enumerate(audios):
            if audio.ndim > 1:
                raise Exception(f"MERaLiON2 only accepts mono channel audio, {i+1}th audio have {audios[0].ndim} channels")
        
        inputs_dict = {}
        
        if audios is not None:
            audio_number_chunks, chunked_audios = self._get_chunked_audios(audios)
            text = self._process_text(text, audio_number_chunks)
            
            audio_inputs = self.feature_extractor(
                chunked_audios, 
                sampling_rate=sampling_rate, 
                return_tensors="pt",
                return_attention_mask=True, 
                padding="max_length", 
                do_normalize=self.do_normalize,
            )
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            inputs_dict.update(audio_inputs)

        text_input = self.tokenizer(
            text=text,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
            padding=padding,
        )   

        inputs_dict["input_ids"] = text_input.input_ids
        inputs_dict["attention_mask"] = text_input.attention_mask

        return BatchFeature(data={**inputs_dict})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))