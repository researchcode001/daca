# agent_generator.py

import json
import os


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def load_agent_meta_prompts(template_file):
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    return templates


def process_template(template_name, template_text, values_data):
    agent_template = {}

    for key, data in values_data.items():
        try:
            data_to_use = data.copy()
            data_to_use['id'] = key
            safe_data = SafeDict(**data_to_use)
            result = template_text.format_map(safe_data)

            print(f"{result}\n{'-' * 50}")

            prompt_key = f"{template_name}_{key}"
            agent_template[prompt_key] = result
        except KeyError as e:
            print(f"Missing value {e}。")
            continue

    return agent_template


def generate_agents(templates_file):
    agent_meta_prompts = load_agent_meta_prompts(templates_file)

    agent_meta_names = ['decomposer', 'polisher', 'assembler']

    all_agent_templates = {}

    for agent_meta_name in agent_meta_names:
        template_text = agent_meta_prompts.get(agent_meta_name)
        if not template_text:
            print(f"template {agent_meta_name} is not in {templates_file}.")
            continue

        values_file = f'data/{agent_meta_name}_value.json'

        if not os.path.exists(values_file):
            print(f"value file {values_file} does not exist，pass template {agent_meta_name} ")
            continue

        with open(values_file, 'r', encoding='utf-8') as f:
            values_data = json.load(f)

        agent_template = process_template(agent_meta_name, template_text, values_data)

        all_agent_templates.update(agent_template)

    return all_agent_templates
