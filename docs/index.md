# ✨ Introduction

Ragas is a library that provides tools to supercharge the evaluation of Large Language Model (LLM) applications. It is designed to help you evaluate your LLM applications with ease and confidence. 



<div class="grid cards" markdown>
- 🚀 **Get Started**

    Install with `pip` and get started with Ragas with these tutorials.

    [:octicons-arrow-right-24: Get Started](getstarted/evals.md)

- 📚 **Core Concepts**

    In depth explanation and discussion of the concepts and working of different features available in Ragas.

    [:octicons-arrow-right-24: Core Concepts](./concepts/index.md)

- 🛠️ **How-to Guides**

    Practical guides to help you achieve a specific goals. Take a look at these
    guides to learn how to use Ragas to solve real-world problems.

    [:octicons-arrow-right-24: How-to Guides](./howtos/index.md)

- 📖 **References**

    Technical descriptions of how Ragas classes and methods work.

    [:octicons-arrow-right-24: References](./references/index.md)

</div>





## Frequently Asked Questions

<div class="toggle-list"><span class="arrow">→</span> What is the best open-source model to use?</div>
<div style="display: none;">
    There isn't a single correct answer to this question. With the rapid pace of AI model development, new open-source models are released every week, often claiming to outperform previous versions. The best model for your needs depends largely on your GPU capacity and the type of data you're working with.
    <br><br>
    It's a good idea to explore newer, widely accepted models with strong general capabilities. You can refer to <a href="https://github.com/eugeneyan/open-llms?tab=readme-ov-file#open-llms">this list</a> for available open-source models, their release dates, and fine-tuned variants.
</div>

<div class="toggle-list"><span class="arrow">→</span> Why do NaN values appear in evaluation results?</div>
<div style="display: none;">
    NaN stands for "Not a Number." In ragas evaluation results, NaN can appear for two main reasons:
    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
        <li><strong>JSON Parsing Issue:</strong> The model's output is not JSON-parsable. ragas requires models to output JSON-compatible responses because all prompts are structured using Pydantic. This ensures efficient parsing of LLM outputs.</li>
        <li><strong>Non-Ideal Cases for Scoring:</strong> Certain cases in the sample may not be ideal for scoring. For example, scoring the faithfulness of a response like "I don't know" might not be appropriate.</li>
    </ul>
</div>

<div class="toggle-list"><span class="arrow">→</span> How can I make evaluation results more explainable?</div>
<div style="display: none;">
    The best way is to trace and log your evaluation, then inspect the results using LLM traces. You can follow a detailed example of this process <a href="/howtos/customizations/metrics/tracing/">here</a>.
</div>

<script>
// FAQ
(function() {
    function initFAQ() {
        const toggles = document.querySelectorAll('.toggle-list');
        
        toggles.forEach(toggle => {
            // Remove any existing listeners
            const newToggle = toggle.cloneNode(true);
            toggle.parentNode.replaceChild(newToggle, toggle);
        });
        
        // Re-select after cloning
        const freshToggles = document.querySelectorAll('.toggle-list');
        
        freshToggles.forEach(toggle => {
            const arrow = toggle.querySelector('.arrow');
            const content = toggle.nextElementSibling;
            
            // Initialize as closed
            if (arrow) arrow.innerText = '→';
            if (content) content.style.display = 'none';
            toggle.classList.remove('active');
            
            // Add click listener
            toggle.addEventListener('click', function() {
                const myContent = this.nextElementSibling;
                const myArrow = this.querySelector('.arrow');
                const isOpen = this.classList.contains('active');
                
                // Close all others first
                freshToggles.forEach(other => {
                    const otherContent = other.nextElementSibling;
                    const otherArrow = other.querySelector('.arrow');
                    if (otherContent) otherContent.style.display = 'none';
                    other.classList.remove('active');
                    if (otherArrow) otherArrow.innerText = '→';
                });
                
                // Open this one if it was closed
                if (!isOpen) {
                    if (myContent) myContent.style.display = 'block';
                    this.classList.add('active');
                    if (myArrow) myArrow.innerText = '↓';
                }
            });
        });
    }
    
    // Initialize when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initFAQ();
        });
    } else {
        initFAQ();
    }
})();
</script>

