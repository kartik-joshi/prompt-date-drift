# prompt-date-drift

This harness asks local Ollama models 200 synthetic date-arithmetic and 'what is current' questions, varying only the date stated in the system prompt, and scores every answer in Python. The README reports the answer flip rate and accuracy delta across five system dates, with bootstrap CIs.
