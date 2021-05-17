# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "2.9.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

# Benchmarking
from transformers.benchmark_utils import (
    Frame,
    Memory,
    MemoryState,
    MemorySummary,
    MemoryTrace,
    UsedMemoryState,
    bytes_to_human_readable,
    start_memory_tracing,
    stop_memory_tracing,
)

# Configurations
from transformers.configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig

from transformers.data import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    is_sklearn_available,
    squad_convert_examples_to_features,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)

# Files and general utilities
from transformers.file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)
from transformers.hf_argparser import HfArgumentParser

# Model Cards
from transformers.modelcard import ModelCard


# Pipelines
from transformers.pipelines import (
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    pipeline,
)

# Tokenizers
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


if is_sklearn_available():
    from transformers.data import glue_compute_metrics, xnli_compute_metrics


# Modeling
if is_torch_available():
    from transformers.modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering, apply_chunking_to_forward

    from .modeling_roberta import (
        RobertaForMaskedLM,
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaForMultipleChoice,
        RobertaForTokenClassification,
        RobertaForQuestionAnswering,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    )
    from .modeling_bert import (
        BertPreTrainedModel,
        BertModel,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BertLayer,
    )
    # Optimization
    from transformers.optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    # Trainer
    from transformers.trainer import Trainer, set_seed, torch_distributed_zero_first, EvalPrediction
    from transformers.data.data_collator import DefaultDataCollator, DataCollator, DataCollatorForLanguageModeling
    from transformers.data.datasets import GlueDataset, TextDataset, LineByLineTextDataset, GlueDataTrainingArguments

if not is_tf_available() and not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
