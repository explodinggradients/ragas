def test_load_config(fake_llm, fake_embedding):

    from ragas.config import DemonstrationConfig, InstructionConfig

    inst_config = InstructionConfig(llm=fake_llm)
    demo_config = DemonstrationConfig(embedding=fake_embedding)
    assert inst_config.llm == fake_llm
    assert demo_config.embedding == fake_embedding
