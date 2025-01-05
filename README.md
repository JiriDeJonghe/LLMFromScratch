# Creating a LLM from Scratch

My belief is that everyone can understand LLMs.

LLMs did not come out of nowhere, as many like to believe, but instead are a logical result of decades of incremental building on top of previously available technologies. This is how it often goes: you find a new technology, people use it, people bump into the limitations, and then you try to remove the limitations, which results in new technology and the cycle repeats.

My belief is that everyone is able to understand LLMs if they start from scratch and goes through the path that scientists have gone through as well. At the end they might even get the feeling that they could've invented it themselves, had they been given the time and opportunity. This is my goal for this repo: provide the reader with a guide to be able to understand, explain and create your own LLM from scratch. This repo will incrementally built an LLM starting from the perceptron and gradually built upon its capabilities until we get to LLMs.

In each chapter I will try to built upon the previously acquired knowledge and each section will have three sections:
1. Overview - in which I try to explain the main intuition and concepts behind the topic. I'll introduce - using logical examples - the capabilities and algorithms in an intuitive manner. I will cover the capabilities, but also the limitations and will use these ideas to introduce the next topic. This should be readable by anyone. Minimum mathematical understanding required.
2. Technical Deepdive - in this section I dive deeper into the technical details. I give more formal definitions, try to show relevant links with other works, and in general try to deepen the mathematical understanding behind the algorithms and LLMs in general. I do assume some prior mathematical background or the willingness of the reader to read up on things I assume to be already understood.
3. Implementation - this section is for the people who like to code: with each chapter I will provide my own C code where I implement the concept that I explain. Users that like to implement it themselves can look at this code for guidance. While the code is written in C, it can be translated to any turing complete language of your choosing. I strongly recommend reading the previous two parts before trying to understand the code, unless you know what you're doing.

I do realize that many (smarter) people before me have undertaken the exact same mission. However what I've often found is that they're either outdated, which is of course a by-product of the fast moving field, and I as well won't be able to cover everything there is to learn about these topics. But what I strive to achieve here are two things:
1. Give you, the reader, a solid understanding such that when people are spewing AI related words, that you at least can contextualize and understand them.
2. Provide you with a sense of wonder and hopefully light a fire such that you are willing to embark on the path of AI yourself, and maybe inspire you to give back to the community yourself, like I have been inspired by those before me.

Happy Learning!

## Chapters
### 1 - Perceptron

We start with one of the simplest, yet very important building blocks: the [Perceptron](https://github.com/JiriDeJonghe/LLMFromScratch/blob/main/1%20-%20Perceptron/README.md). Neural networks, transformers, and LLMs would all not be possible was it not for this small building block.

Simply put, a perceptron is an algorithm used for binary classification by estimating a linear function that divides the search space into two, where samples falling on opposite sides have opposite labels. After estimating the linear function, the perceptron can be used for inference to classify unseen samples.

### 2 - Neural Networks

WIP


