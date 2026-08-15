from typing import Any
from .utils.prompt import SystemPrompt

def format_train_data(data: Any) -> dict[str, str]:
    formatted = {}
    formatted['messages'] = [
        {
            'role': 'user',
            'content': {
                'type': 'text',
                'text': f'Question:{data['question']}'
            }
        },
        {
            'role': 'assistant',
            'content': {
                'type': 'text',
                'text': f'Answer:{data['reference_answer']}'
            },
        }
    ]
    return formatted

def format_test_data(data: Any) -> dict[str, str]:
    formatted = {}
    formatted['messages'] = [
        {
            'role': 'system',
            'content': {
                'type': 'text',
                'text': SystemPrompt
            }
        },
        {
            'role': 'user',
            'content': {
                'type': 'text',
                'text': f'Question:{data['question']}'
            }
        }
    ]
    return formatted
