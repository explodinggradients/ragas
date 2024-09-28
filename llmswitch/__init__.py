from mkdocs.plugins import BasePlugin

print("LLMSwitchPlugin module loaded")

class LLMSwitchPlugin(BasePlugin):
    def on_page_markdown(self, markdown, page, config, files):
        print("LLMSwitchPlugin processing page")
        return markdown.replace('{% llm_tabs %}', '<div class="llm-tabs">').replace('{% endllm_tabs %}', '</div>')