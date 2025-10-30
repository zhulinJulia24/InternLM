import pytest
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

prompts = ['你好', "what's your name"]

def assert_model(response):
    assert len(response) != 0
    assert 'UNUSED_TOKEN' not in response
    assert 'Mynameis' not in response
    assert 'Iama' not in response


class TestChat:
    @pytest.mark.parametrize('model_name', ['internlm/Intern-S1'])
    @pytest.mark.parametrize('enable_thinking', [True, False])
    @pytest.mark.gpu_num_8
    @pytest.mark.interns1
    def test_demo_default_gpu8(self, model_name, enable_thinking):
        test_s1_chat_demo(model_name, enable_thinking)

    @pytest.mark.parametrize('model_name', ['internlm/Intern-S1-mini'])
    @pytest.mark.parametrize('enable_thinking', [True, False])
    @pytest.mark.gpu_num_1
    @pytest.mark.interns1
    def test_demo_default_gpu1(self, model_name, enable_thinking):
        test_s1_chat_demo(model_name, enable_thinking)


def test_s1_chat_demo(model_name, enable_thinking):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'tell me about an interesting physical phenomenon.'}]}]

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to( model.device, dtype=torch.bfloat16)

    generate_ids = model.generate(**inputs, max_new_tokens=32768)
    decoded_output = processor.decode(generate_ids[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True)
    print(decoded_output)
    assert_model(decoded_output)
