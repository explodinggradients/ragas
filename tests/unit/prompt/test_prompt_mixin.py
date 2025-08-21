import pytest

from ragas.testset.synthesizers.multi_hop import MultiHopAbstractQuerySynthesizer


def test_prompt_save_load(tmp_path, fake_llm):
    synth = MultiHopAbstractQuerySynthesizer(llm=fake_llm)
    synth_prompts = synth.get_prompts()
    synth.save_prompts(tmp_path)
    loaded_prompts = synth.load_prompts(tmp_path)
    assert len(synth_prompts) == len(loaded_prompts)
    for name, prompt in synth_prompts.items():
        assert name in loaded_prompts
        assert prompt == loaded_prompts[name]


@pytest.mark.asyncio
async def test_prompt_save_adapt_load(tmp_path, fake_llm):
    synth = MultiHopAbstractQuerySynthesizer(llm=fake_llm)

    # patch adapt_prompts
    async def adapt_prompts_patched(self, language, llm):
        for prompt in self.get_prompts().values():
            prompt.instruction = "test"
            prompt.language = language
        return self.get_prompts()

    synth.adapt_prompts = adapt_prompts_patched.__get__(synth)

    # adapt prompts
    original_prompts = synth.get_prompts()
    adapted_prompts = await synth.adapt_prompts("spanish", fake_llm)
    synth.set_prompts(**adapted_prompts)

    # save n load
    synth.save_prompts(tmp_path)
    loaded_prompts = synth.load_prompts(tmp_path, language="spanish")

    # check conditions
    assert len(adapted_prompts) == len(loaded_prompts)
    for name, adapted_prompt in adapted_prompts.items():
        assert name in loaded_prompts
        assert name in original_prompts

        loaded_prompt = loaded_prompts[name]
        assert adapted_prompt.instruction == loaded_prompt.instruction
        assert adapted_prompt.language == loaded_prompt.language
        assert adapted_prompt == loaded_prompt
