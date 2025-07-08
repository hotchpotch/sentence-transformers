"""
Unit tests for PruningEncoder modes (reranking_pruning and pruning_only).
"""

import pytest
import torch
import numpy as np
from datasets import load_dataset

from sentence_transformers.pruning import (
    PruningEncoder, PruningTrainer, PruningLoss, PruningDataCollator
)
from sentence_transformers.pruning.data_structures import (
    RerankingPruningOutput, PruningOnlyOutput
)


class TestPruningModes:
    """Test both pruning modes."""
    
    @pytest.fixture
    def minimal_dataset(self):
        """Load minimal dataset for testing."""
        dataset = load_dataset(
            'hotchpotch/wip-query-context-pruner-with-teacher-scores',
            'ja-minimal'
        )
        return dataset['train'].select(range(4))  # Use only 4 samples for speed
    
    def test_reranking_pruning_mode_init(self):
        """Test initialization of reranking_pruning mode."""
        model = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning",
            device="cpu"
        )
        
        assert model.mode == "reranking_pruning"
        assert model.ranking_model is not None
        assert model.encoder is None
        assert model.pruning_head is not None
    
    def test_pruning_only_mode_init(self):
        """Test initialization of pruning_only mode."""
        model = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu",
            pruning_config={
                "hidden_size": 256,  # Match ruri-v3-30m
                "dropout": 0.1
            }
        )
        
        assert model.mode == "pruning_only"
        assert model.ranking_model is None
        assert model.encoder is not None
        assert model.pruning_head is not None
    
    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            PruningEncoder(
                model_name_or_path="cl-nagoya/ruri-v3-30m",
                mode="invalid_mode"
            )
    
    def test_reranking_pruning_forward(self):
        """Test forward pass in reranking_pruning mode."""
        model = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning",
            device="cpu"
        )
        
        # Create dummy inputs
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        assert "ranking_logits" in outputs
        assert "pruning_logits" in outputs
        assert "hidden_states" in outputs
        assert outputs["ranking_logits"].shape == (batch_size, 1)
        assert outputs["pruning_logits"].shape == (batch_size, seq_len, 2)
    
    def test_pruning_only_forward(self):
        """Test forward pass in pruning_only mode."""
        model = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu",
            pruning_config={"hidden_size": 256}
        )
        
        # Create dummy inputs
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        assert "ranking_logits" not in outputs
        assert "pruning_logits" in outputs
        assert "hidden_states" in outputs
        assert outputs["pruning_logits"].shape == (batch_size, seq_len, 2)
    
    def test_data_collator_reranking_pruning(self, minimal_dataset):
        """Test data collator in reranking_pruning mode."""
        tokenizer = PruningEncoder(
            "hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning",
            device="cpu"
        ).tokenizer
        
        collator = PruningDataCollator(
            tokenizer=tokenizer,
            mode="reranking_pruning"
        )
        
        batch = collator(minimal_dataset)
        
        assert "ranking_targets" in batch["labels"]
        assert "pruning_labels" in batch["labels"]
        assert isinstance(batch["labels"]["ranking_targets"], torch.Tensor)
    
    def test_data_collator_pruning_only(self, minimal_dataset):
        """Test data collator in pruning_only mode."""
        tokenizer = PruningEncoder(
            "cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu"
        ).tokenizer
        
        collator = PruningDataCollator(
            tokenizer=tokenizer,
            mode="pruning_only"
        )
        
        batch = collator(minimal_dataset)
        
        assert "ranking_targets" not in batch["labels"]
        assert "pruning_labels" in batch["labels"]
    
    def test_loss_computation_reranking_pruning(self, minimal_dataset):
        """Test loss computation in reranking_pruning mode."""
        model = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning",
            device="cpu"
        )
        
        collator = PruningDataCollator(
            tokenizer=model.tokenizer,
            mode="reranking_pruning"
        )
        
        loss_fn = PruningLoss(
            model=model,
            mode="reranking_pruning"
        )
        
        batch = collator(minimal_dataset)
        loss = loss_fn(batch["sentence_features"], batch["labels"])
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_loss_computation_pruning_only(self, minimal_dataset):
        """Test loss computation in pruning_only mode."""
        model = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu",
            pruning_config={"hidden_size": 256}
        )
        
        collator = PruningDataCollator(
            tokenizer=model.tokenizer,
            mode="pruning_only"
        )
        
        loss_fn = PruningLoss(
            model=model,
            mode="pruning_only"
        )
        
        batch = collator(minimal_dataset)
        loss = loss_fn(batch["sentence_features"], batch["labels"])
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_predict_error_in_pruning_only(self):
        """Test that predict without pruning raises error in pruning_only mode."""
        model = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu"
        )
        
        with pytest.raises(ValueError, match="pruning_only mode"):
            model.predict(("query", "document"), apply_pruning=False)
    
    def test_prune_texts_method(self):
        """Test prune_texts method for pruning_only mode."""
        model = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu",
            pruning_config={"hidden_size": 256}
        )
        
        queries = ["これは質問です", "別の質問です"]
        texts = ["これは文書です。関連する内容。", "これは別の文書です。"]
        
        results = model.prune_texts(queries, texts, threshold=0.5)
        
        assert len(results) == 2
        assert all("pruned_text" in r for r in results)
        assert all("kept_ratio" in r for r in results)
        assert all(0 <= r["kept_ratio"] <= 1 for r in results)
    
    def test_save_load_reranking_pruning(self, tmp_path):
        """Test save and load functionality for reranking_pruning mode."""
        model1 = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning",
            device="cpu"
        )
        
        save_path = tmp_path / "test_model"
        model1.save_pretrained(save_path)
        
        model2 = PruningEncoder.from_pretrained(save_path, device="cpu")
        
        assert model2.mode == "reranking_pruning"
        assert model2.ranking_model is not None
        assert model2.encoder is None
    
    def test_save_load_pruning_only(self, tmp_path):
        """Test save and load functionality for pruning_only mode."""
        model1 = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu"
        )
        
        save_path = tmp_path / "test_model"
        model1.save_pretrained(save_path)
        
        model2 = PruningEncoder.from_pretrained(save_path, device="cpu")
        
        assert model2.mode == "pruning_only"
        assert model2.ranking_model is None
        assert model2.encoder is not None
    def test_output_types(self):
        """Test that correct output types are returned based on mode."""
        # Reranking + Pruning mode
        model1 = PruningEncoder(
            model_name_or_path="hotchpotch/japanese-reranker-xsmall-v2",
            mode="reranking_pruning",
            device="cpu"
        )
        
        output1 = model1.predict(
            ("query", "document"),
            apply_pruning=True
        )
        assert isinstance(output1, RerankingPruningOutput), \
            f"Expected RerankingPruningOutput, got {type(output1)}"
        
        # Pruning-only mode
        model2 = PruningEncoder(
            model_name_or_path="cl-nagoya/ruri-v3-30m",
            mode="pruning_only",
            device="cpu",
            pruning_config={"hidden_size": 256}
        )
        
        output2 = model2.predict(
            ("query", "document"),
            apply_pruning=True
        )
        assert isinstance(output2, PruningOnlyOutput), \
            f"Expected PruningOnlyOutput, got {type(output2)}"
