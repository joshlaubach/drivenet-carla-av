"""Smoke tests verifying all agents import and those that don't need CARLA can instantiate."""

import pytest

pytest.importorskip("carla", reason="CARLA Python API not available")


def test_import_all_agents() -> None:
    """All 5 agents should be importable from their respective modules."""
    from src.agents.causal_agent import CausalAnalysisAgent
    from src.agents.collection_agent import DataCollectionAgent
    from src.agents.eval_agent import EvaluationAgent
    from src.agents.ppo_agent import PPOAgent
    from src.agents.training_agent import BehaviorCloningAgent
    assert DataCollectionAgent is not None
    assert BehaviorCloningAgent is not None
    assert PPOAgent is not None
    assert EvaluationAgent is not None
    assert CausalAnalysisAgent is not None


def test_causal_agent_instantiates() -> None:
    """CausalAnalysisAgent does not require CARLA in __init__."""
    from src.agents.causal_agent import CausalAnalysisAgent
    agent = CausalAnalysisAgent()
    assert agent.cfg is not None
    assert len(agent.treatments) == 6  # 3 sensor comparisons + rain, night, style


def test_bc_agent_instantiates() -> None:
    """BehaviorCloningAgent does not require CARLA in __init__."""
    from src.agents.training_agent import BehaviorCloningAgent
    agent = BehaviorCloningAgent()
    assert agent.cfg is not None
    assert agent.device is not None
