from mkdocs.plugins import BasePlugin

class LLMSwitchPlugin(BasePlugin):
    def on_page_markdown(self, markdown, page, config, files):
        return markdown.replace('{% llm_tabs %}', '<div class="llm-tabs">').replace('{% endllm_tabs %}', '</div>')