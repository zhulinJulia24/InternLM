import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

prompts = ['你好', "what's your name"]


def assert_model(response):
    assert len(response) != 0
    assert 'UNUSED_TOKEN' not in response
    assert 'Mynameis' not in response
    assert 'Iama' not in response


@pytest.fixture(scope='function', autouse=True)
def device_setup(request, worker_id):
    if request.node.get_closest_marker('gpu_num_8'):
        tp_num = 8
    elif request.node.get_closest_marker('gpu_num_4'):
        tp_num = 4
    elif request.node.get_closest_marker('gpu_num_2'):
        tp_num = 2
    elif request.node.get_closest_marker('gpu_num_1'):
        tp_num = 1
    else:
        tp_num = 1
    
    set_device_env_variable(worker_id, tp_num)
    yield
    unset_device_env_variable()


@pytest.mark.parametrize('model_name', ['internlm/Intern-S1'])
@pytest.mark.parametrize('enable_thinking', [True, False])
@pytest.mark.gpu_num_8
@pytest.mark.interns1
def test_demo_default_gpu8(model_name, enable_thinking, device_setup):
    test_s1_chat_text_demo(model_name, enable_thinking)
    test_s1_chat_image_demo(model_name, enable_thinking)
    test_s1_chat_video_demo(model_name, enable_thinking)

@pytest.mark.parametrize('model_name', ['internlm/Intern-S1-mini'])
@pytest.mark.parametrize('enable_thinking', [True, False])
@pytest.mark.gpu_num_1
@pytest.mark.interns1
def test_demo_default_gpu1(model_name, enable_thinking, device_setup):
    test_s1_chat_text_demo(model_name, enable_thinking)
    test_s1_chat_image_demo(model_name, enable_thinking)
    test_s1_chat_video_demo(model_name, enable_thinking)

def test_s1_chat_text_demo(model_name, enable_thinking):
    processor = AutoProcessor.from_pretrained(model_name,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto',
                                                 torch_dtype='auto',
                                                 trust_remote_code=True)
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

        inputs = processor.apply_chat_template(messages,
                                               add_generation_prompt=True,
                                               enable_thinking=enable_thinking,
                                               tokenize=True,
                                               return_dict=True,
                                               return_tensors='pt').to(
                                                   model.device,
                                                   dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        decoded_output = processor.decode(
            generate_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        assert 'physical phenomenon' in decoded_output.lower() or '物理现象' in decoded_output, decoded_output
        assert_model(decoded_output)

def test_s1_chat_image_demo(model_name, enable_thinking):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

    prompts = [
        'Please describe the image explicitly.', '请描述这个图像。'
    ]
    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        assert 'physical phenomenon' in decoded_output.lower() or '物理现象' in decoded_output, decoded_output
        assert_model(decoded_output)


def test_s1_chat_video_demo(model_name, enable_thinking):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

    prompts = [
        'What type of shot is the man performing?', '这个人正在进行什么类型的击球？'
    ]
    for prompt in prompts:
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        inputs = processor.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                video_load_backend="decord",
                tokenize=True,
                return_dict=True,
            ).to(model.device, dtype=torch.float16)

        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        assert 'physical phenomenon' in decoded_output.lower() or '物理现象' in decoded_output, decoded_output
        assert_model(decoded_output)

def get_cuda_prefix_by_workerid(worker_id, tp_num: int = 1):
    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if cuda_id is None or 'gw' not in worker_id:
        return None
    else:
        device_type = os.environ.get('DEVICE', 'cuda')
        if device_type == 'ascend':
            return 'ASCEND_RT_VISIBLE_DEVICES=' + cuda_id
        else:
            return 'CUDA_VISIBLE_DEVICES=' + cuda_id

def get_cuda_id_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return ','.join([str(cuda_num), str(cuda_num + 1)])
        elif tp_num == 4:
            cuda_num = int(worker_id.replace('gw', '')) * 4
            return ','.join([
                str(cuda_num),
                str(cuda_num + 1),
                str(cuda_num + 2),
                str(cuda_num + 3)
            ])


def set_device_env_variable(worker_id, tp_num: int = 1):
    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if cuda_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id


def unset_device_env_variable():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
