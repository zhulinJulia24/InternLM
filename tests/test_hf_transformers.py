import pytest
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def assert_model(response):
    assert len(response) != 0
    assert 'UNUSED_TOKEN' not in response
    assert 'Mynameis' not in response
    assert 'Iama' not in response


@pytest.mark.parametrize('model_name',
                         ['internlm/Intern-S1', 'internlm/Intern-S1-mini'])
@pytest.mark.parametrize('enable_thinking', [True, False])
@pytest.mark.gpu_num_8
@pytest.mark.interns1
def test_demo_default(model_name, enable_thinking):
    autoprocessor, model = get_processor(model_name)
    test_s1_chat_text_demo(autoprocessor, model, enable_thinking)
    test_s1_chat_image_demo(autoprocessor, model, enable_thinking)
    test_s1_chat_video_demo(autoprocessor, model, enable_thinking)


def get_processor(model_name):
    autoprocessor = AutoProcessor.from_pretrained(model_name,
                                                  trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto',
                                                 torch_dtype='auto',
                                                 trust_remote_code=True)
    return autoprocessor, model


def test_s1_chat_text_demo(autoprocessor, model, enable_thinking):
    prompts = [
        'tell me about an interesting physical phenomenon.', '请给我讲一个有趣的物理现象'
    ]
    for prompt in prompts:
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': prompt
            }]
        }]

        inputs = autoprocessor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=True,
            return_tensors='pt').to(model.device, dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        decoded_output = autoprocessor.decode(
            generate_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        assert 'physical phenomenon' in decoded_output.lower(
        ) or '物理现象' in decoded_output, decoded_output
        assert_model(decoded_output)


def test_s1_chat_image_demo(autoprocessor, model, enable_thinking):
    prompts = ['Please describe the image explicitly.', '请描述这个图像。']
    for prompt in prompts:
        messages = [{
            'role':
            'user',
            'content': [
                {
                    'type': 'image',
                    'url':
                    'http://images.cocodataset.org/val2017/000000039769.jpg'
                },
                {
                    'type': 'text',
                    'text': prompt
                },
            ],
        }]

        inputs = autoprocessor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=True,
            return_tensors='pt').to(model.device, dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        decoded_output = autoprocessor.decode(
            generate_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        assert 'cat' in decoded_output.lower(
        ) or '猫' in decoded_output, decoded_output
        assert_model(decoded_output)


def test_s1_chat_video_demo(autoprocessor, model, enable_thinking):
    prompts = ['What type of shot is the man performing?', '这个人正在进行什么类型的击球？']
    for prompt in prompts:
        messages = [{
            'role':
            'user',
            'content': [
                {
                    'type':
                    'video',
                    'url':
                    'https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4',  # noqa: E501
                },
                {
                    'type': 'text',
                    'text': prompt
                },
            ],
        }]

        inputs = autoprocessor.apply_chat_template(
            messages,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            video_load_backend='decord',
            tokenize=True,
            return_dict=True,
        ).to(model.device, dtype=torch.float16)

        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        decoded_output = autoprocessor.decode(
            generate_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        assert 'physical phenomenon' in decoded_output.lower(
        ) or '物理现象' in decoded_output, decoded_output
        assert_model(decoded_output)
