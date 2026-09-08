# Understand Gemma-4-31b sampling mechanism

## Context

In the past session, we further investigated the performance of gemma-4-31b on the Pylyshyn task, and found that it still displays a fairly robust ability to perform the Pylyshyn task, even over and above the qwen3.6-plus model. Given that the gemma-4-31b model only consumes a fraction of the tokens it should and how the input token amount doesn't change when the number of frames varies, it remains an open question as to what the sampling mechanism is.

## Probing the sampling mechanism

Let's see if we can get the number of input token to change with a small number of frames. Can we work up from 1 to 15 to see if the number of input tokens increases and saturates? Let's use the openrouter api and hand construct some stimuli.
