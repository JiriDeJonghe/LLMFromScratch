r Perceptron

## History

The disruption caused by LLMs (Large Language Models) might seem to have come out of nowhere, but the core building block can be traced all the way back to 1957, when the perceptron was first conceived by Frank Rosenblatt. He thought of the perceptron as a computational representation of a neuron as found in the human brain. Initially, the perceptron was implemented as a hardware machine, as opposed to a general algorithm which we're going to discuss here.

And thus, it is safe to say that the idea has been around for a long time, but if that's the case, why has the GenAI revolution only starting recently? And what exactly is a perceptron, and more importantly, what can it be used for?

## Intuition
### Making decisions: to walk or not to walk

Imagine you're deciding whether to go for a walk or not. Your willingness to go depends on two factors:
1. The weather is good
2. You have time 
We can decide to represent the two factors via a **binary encoding**, which is just a fancy way of saying that we use 0 and 1 to represent whether the statements are true or not:

| Expression | Encoding |
| --- | --- |
|The weather is good | $x_{weather} = 1$ |
| The weather is NOT good | $x_{weather} = 0$ |
| You have time | $x_{time} = 1$ |
| You do not have time | $x_{time} = 0$ |

So if $x_{weather} = 1$, then that means that the weather is good. If $x_{weather} = 0$, then the weather is not good. Since your willingness to go for a walk depends on these two statements we can define the following equation: $$y_{willingness} = f(x_{weather}, x_{time})$$, with $f$ representing a function that we don't know yet. To explain it in natural language, we define our **output** $y_{willingness}$ based on a function that takes $x_{weather}$ and $x_{time}$ as **inputs**.

And this is an important observation and this observation can be made for any AI model. Any AI model, in any shape or form, takes one or more inputs and produces one or more outputs (usually indicated by $x$ and $y$ respectively).

Suppose now that the previously unknown function is now just simply the sum, the equation then becomes very simple: $$y_{willingness} = x_{weather} + x_{time}$$

By varying our inputs, we can get different values for our output:

$x_{weather}$ | $x_{time}$ | $y_{willingness}$ 
--- | --- | ---
0 | 0 | 0
0 | 1 | 1
1 | 0 | 1
1 | 1 | 2

Notice that there are three possible values for $y_{willingness}$: 0, 1, and 2. In order to interpret these values, we need to take a step back, and ask ourselves the following question: "when do we want to go for a walk?". Suppose that you only want to go for a walk only if the weather is both good and you have time. In this case your willingness to go for a walk is the highest it can be. This translates to the last row in the table above, when $y_{willingness} = 2$. If the weather is good but you have no time, or if you have time but the weather is not good, you do not want to go for a walk. This corresponds to row two and three, when $y_{willingness} = 1$. Evidently, if the weather is not good and you don't have time to go for a walk, then you don't go for a walk. We summarize:


$y_{willingness}$ | Go for a walk?
--- | --- 
0 | No 
1 | No
2 | Yes

Remember when we did the binary encoding for the inputs? A similar encoding can be made for whether to go for a walk:

Expression | Encoding
--- | ---
You go for a walk | $y_{walk} = 1$
You do not go for a walk | $y_{walk} = 0$

Unfortunately, we only have two values here, as opposed to the previously resulted three. It seems that simply taking the sum will not get us to where we need be. So, we add a very small change, we define the **Heaviside step function**, which maps a value to either 0 or 1. Let's say that every value below 2 gets mapped to 0, and every value above or equal to 2 gets mapped to 1.

We summarize everything up till now:

Good Weather? | Free time? | $x_{weather}$ | $x_{time}$ | $y_{willingness}$ | $y_{walk}$
--- | --- | --- | --- | --- | ---
No | No | 0 | 0 | 0 | 0
No | Yes | 0 | 1 | 1 | 0
Yes | No | 1 | 0 | 1 | 0
Yes | Yes | 1 | 1 | 2 | 1

And we're done! Indeed, to recap we took the following steps:
1. We encoded the inputs as 0 or 1 to represent whether the weather was good and if you have free time
2. We took the sum of the inputs
3. We mapped the resulting sum to 0 or 1

### Taking into account importance

Now suppose that you attach more importance to one of the two conditions, suppose you still want to go for a walk if the weather is good, even though you don't have the time to do so. This cannot be represented in the model above. Suppose you would create an updated decision table:

$y_{willingness}$ | Go for a walk?
--- | --- 
0 | No 
1 | Yes
2 | Yes

Indeed, this updated table will indicate that we want to go for a walk if the weather is good, despite not having the time. However, this has an unintended consequence: according to the table, you will also go for a walk if you have time but the weather is not good. It becomes clear once we write out the entire table:

Good Weather? | Free time? | $x_{weather}$ | $x_{time}$ | $y_{willingness}$ | $y_{walk}$
--- | --- | --- | --- | --- | ---
No | No | 0 | 0 | 0 | 0
No | Yes | 0 | 1 | 1 | **1**
Yes | No | 1 | 0 | 1 | **1**
Yes | Yes | 1 | 1 | 2 | 1

Now this is an issue, because maybe the weather being good is very important to you, and you don't want to go for a walk if you have time but the weather is not good. Suppose we represent the importance of having free time as 1, in that case, the importance of the weather being good is higher, let's say 2. These values are called **weights**, which represents to what extend an input is important to take into account when deciding on your output. The weights (importance) can be represented as follows $w_{weather} = 2$ and $w_{time} = 1$.

Now, let's recall what we did before to get to our output value, we took the sum of the different inputs, and put it through a function that mapped it to 0 or 1. Including weights in this process is actually relatively simple: instead of taking the normal sum, we take the **weighted sum**, which is simply the sum of the inputs multiplied by their importance: $$y_{willingness} = w_{weather}*x_{weather} + w_{time}*x_{time} = 2*x_{weather} + 1*x_{time}$

Using this newly obtained formula we can update the tables:

Good Weather? | Free time? | $x_{weather}$ | $x_{time}$ | $y_{willingness}$ 
--- | --- | --- | --- | --- 
No | No | 0 | 0 | 0 
No | Yes | 0 | 1 | 1
Yes | No | 1 | 0 | 2 (= 2*1 + 1*0)
Yes | Yes | 1 | 1 | 3 (= 2*1 + 1*1)

But, now we have increased the number of possible values for $y_{willingness}$, so how can we map them now to 0 and 1? Turns out we don't need to change anything, recall from before: every value below 2 gets mapped to 0, and every value above or equal to 2 gets mapped to 1. This results in the following table:

Good Weather? | Free time? | $x_{weather}$ | $x_{time}$ | $y_{willingness}$ | $y_{walk}$
--- | --- | --- | --- | --- | ---
No | No | 0 | 0 | 0 | 0
No | Yes | 0 | 1 | 1 | **0**
Yes | No | 1 | 0 | 2 | **1**
Yes | Yes | 1 | 1 | 3 | 1

One small caveat, and if this is not immediately clear, you can continue, but for completeness sake, it is required to cover this: the **Heaviside step function** actually maps negative values to 0 and positive values (0 included) to 1 (as opposed to values smaller or bigger then 2). To help with this shortcoming, the perceptron also has, besides the input, a thing called the **bias**. The bias is taken into account in the weighted sum and decides which values get mapped to 0 and which ones to 1. We update the weighted sum equation: $$y_{willingness} = w_{weather}*x_{weather} + w_{time}*x_{time} + b,$$ with $b$ the bias.

To prove correctness we can fill in the equation. Suppose $b = -2$, the formula then becomes:$$y_{willingness} = 2*x_{weather} + x_{time} - 2$$. Let's consider our table again:

Good Weather? | Free time? | $x_{weather}$ | $x_{time}$ | $y_{willingness}$ 
--- | --- | --- | --- | --- 
No | No | 0 | 0 | -2 (= 0+0-2) 
No | Yes | 0 | 1 | -1 (= 0+1-2)
Yes | No | 1 | 0 | 0 (= 2+0-2)
Yes | Yes | 1 | 1 | 1 (= 2+1-2)

Congratulations, you now understand the steps a perceptron goes through to perform **inference**, which is the action of deciding the output when given the inputs. Consider Figure 1 for a visual representation of the perceptron. You'll see that we have covered each part of the decision process. The inputs are coming from the left annotated with the weights. TODO

[IMAGE 1]

### Learning from your mistakes

You might've noticed that up until this points the model hasn't actually learned anything. It simply takes the inputs and produces an output based on a sum the heaviside step function. You'd be completely right, up to this point we have only talked about inference, but we haven't talked about where the model gets its parameters to make this inference. I've provided you with the weights and bias that makes sense for the story, and if we would be able to deduce all these parameters ourselves in any scenario, then there would be no need to use AI, since we can perfectly model each decision that we want to make.

This is where the learning part comes in, called **training**. This might seem a little daunting (and I will spare you the technical details), but it turns out, this part too is actually quite intuitive. To show this, let's illustrate this by taking the previous example. 

Now suppose that your friend somehow convinced you to go on a walk even though the weather was not that nice. And, despite your unwillingness to go, suprisingly enough, you very much enjoyed the walk. This is a new experience for you and you decide to reflect about your previous decision making, it's a time for learning. You realise that in the past you might've attached too much importance on the weather aspect during your decision making, because, it turns out that walks can still be very enjoyable despite weather! 

Let's consider one more example, imagine you're going on a walk because the weather is nice, despite having no time to do so since you still have a lot of other chores you need to do at home. You very much enjoy the walk, but afterwards do not have the time to fullfill your chores, and you once again reflect, maybe having free time should be more important?

Both of these examples show how an experience can influence the way you reason about your surroundings and your decision making. Next time, your willingness to go for a walk might be lower if you don't have free time, it becomes a more important requirement. Looking back at our perceptron, we realise we already have a way of indicating our importance: the weights.

And there you have it, this is exactly what the training of a perceptron entails: based on new experiences, we update the parameters of the models (the weights and the bias), to better reflect the new reality. So when training the perceptron, we provide it with combinations of inputs and the resulting decision that the perceptron should make. The perceptron will slowly adjust its weights and bias in such a way that it is best able to make these decisions. After the model has been trained, we can use it for inference by only providing outputs to the perceptron, the perceptron will decide on the result. To summarize the difference between training and inference:

Task | Description | Requirements
--- | --- | ---
Training | This is where the model learns from examples. The model will adjusts its parameters based on the examples provided in order to optimally produce outputs for this data. This is also called **fitting** the data. | Examples with inputs and outputs provided
Inference | Can only be done after training. This is where the model produces the output of a new occurence. This is also called **predicting** the data. | Only the inputs of the occurence

### The Perceptron

Finally, we've arrived at the points where we can define the perceptron by generalizing. The perceptron is...
1. ...a **classifer**: it assigns the occurence to a category (to walk or not to walk in the previous example). 
2. ...a **binary** classifier: there are only two categories to assign to. An occurence can only belong to one of the two, there are no other options.

A few remarks:
- Up till this point we have considered the inputs to be 0 or 1. This is not necessarily the case as we will see in the next example. The inputs can be any type of number.
- In this post we have limited the number of inputs for the perceptron to two, for clarity's sake, but there's no limit on the number of inputs it can have.
For example, we could add a third consideration in the previous example: how many people will be joining you on the walk?

### Visualization

Let's take a look at a more complex example now: suppose we want to detect whether a bank transfer is fraudulent or legit. This is a binary classification problem: a transfer is either fraudulent or it is not. There are no other options, there is no grey zone, and a transfer cannot be a "little" fraudulent. And so, it seems that we can use the perceptron to classify each transfer into one of the two categories.

Now, suppose that there are two factors that contribute to whether a transfer is fraudulent: the amount that has been transfered and the time the transfer was initiated. Of course, this is as strong of an over-simplification that can be made, but it will suffice for now. Let's now make an even stronger assumption, namely that a transfer is fraudulent if and only if it was done both late at night, (e.g., between 12 and 6 AM) and if the amount transferred is for an amount bigger then 10,000 USD. The following shows a few occurences:

Example | Time of Transfer | Amount (USD) | Fraudulent?
--- | --- | --- | ---
1 | 8 PM | 325 | No
2 | 5 AM | 3485 | No
3 | 1 PM | 10329 | No
4 | 3 AM | 11399 | Yes

Only the fourth transfer is fraudulent, since it is done at 3 AM and it is an amount of more than 10000 USD. The other three either only satsify one of the conditions, or not a single one.

Consider Image 1. It shows legitimate and fradulent transfers, plotted out in a graph: on the x-axis we find the time of execution, on the y-axis we put the amount of money transfered. The fraudulent transfers are indicated in red, while the legitimate are indicated in blue. Imagine drawing a straight line to divide the red and blue dots into two, or at least try to limit the amount of wrongly placed dots (blue dots on the side of reds, or red dots on the side of blues).

[IMAGE 2 HERE]

This is exactly what the perceptron tries to do as well: it tries to find the linear function that best divides the datapoints into two categories, such that the amounts of misclassifications is minimized. If you've read the first section, recall the weighted sum of before: the function that will be drawn is defined by the weighted sum: 
$$w_{amount}*x_{amount} + w_{time}*x_{time} + b = 0$$

The approach the perceptron takes to find the (or 'a', if multiple solutions can be found) function that best classifies the data is quite simple. First, start with a randomly generated line. After that, the goal is to update the parameters of this line to achieve better performance.

But what does it mean to have better performance? To measure the quality of a model, we use **metrics**. Metrics are dependent on what the use-case is of the model, but in this case we can keep it very simple: 
$$Accuracy = \frac{Correctly classified datapoints}{Total amount of datapoints}.$$ 
To conclude: a perceptron tries to find the parameters of a linear function, that separates the examples into two areas, one for each category. At inference time, we take a look at the inputs of this occurence, and look at which area it falls into. It will get assigned to that category.

Figure 3 illustrates how a straight line creates two areas.

[FIGURE 3 here]

And that's all there is to it! Figure 4 shows the evolution of the line to fit the data during training:

[FIGURE 4 HERE]

Our final model achieves an *accuracy* of 97%, meaning that three cases are misclassified. This is pretty good, but do remember that we have simplified the reality until it fit the narrative I wanted to tell. Unfortunately, this is where the perceptron starts showing its limitations...

There are three main limitations with the perceptron:
1. **Limited to two classes**: as mentioned previously, the perceptron is limited to two classes. This is a very severel limitation, because often we want to be able to get more information. Maybe you've decided that a transfer is fraudulent, but what type of fraud is happening here? Is the person actively committing fraud, or are they being tricked by a scammer?
2. **Not every decision can be made by drawing a line**: we showed that the perceptron is only able to define a straight line to differentiate between the two classes, however, it turns out that it is very rarely the case that data can be separated this easily. More complex functions will need to be constructed to be able to tackle these problems.
3. **Misclassification Importance**: when training the perceptron, it makes no difference between the importance of a misclassification. In our fraud example, a fraudulent transfer that has been missed because it was considered legit is much more severe than a legit transfer that has been classified as fraud. In the former case, you potentially lose a lot of money, while in the latter, you will have to manually investigate an extra case. Extra metrics have been defined to catch this difference, and more complex models can optimize for these metrics, instead of accuracy, which can lead to wrong interpretation.

## Technical Deepdive

In this section I assume the reader to have a somewhat stronger background in mathematics. The underlying mathematics are actually not that complex, but they are not necessarily required to grasp the full picture. Feel free to skim over this section or skip it altogether. However, if aiming to imlement the perceptron yourself, it is recommended to have a deep understanding of the underlying algorithms.

### Updating weights

As mentioned before, the perceptron works by taking the weighted sum over the inputs using the weights: 
$$
y = w_1x_1 + w_2x_2 + \ldots + w_ix_i + b
\sum_{i} w_ix_i + b
$$

Or alternatively, let $$W = \begin{bmatrix} w_1 \\ w_2 \\ \ldots \\ w_i$$
and $$x = \begin{bmatrix} x_1 \\ x_2 \\ \ldots \\ x_i$$
Then we get $$weighted_sum = W^\intercal x+ b.$$ 

Furthermore, we define the following:
$w_i^{t+1}$: the next value of weight $i$
$w_i^{t}$: the current value of weight $i$
$y$: the target value
$\hat{y}$: the result of the perceptron
$r$: **learning rate**, value between 0 and 1 that indicates how much the parameters should be changed for each occurence
To update the weights we define the following:
$$
w_i^{t+1} = w_i^t + r*(y - \hat{y})*x_i,
b^{t+1} = b^{t} + r*(y - \hat{y}).
$$
Notice that if the perceptron has predicted the output correctly, than $(y - \hat{y}) = 0$ and the weights and bias remain unchanged.

It is very common to train a model multiple times on the same dataset. A complete pass of the dataset is called an **epoch**. This is done because usually one pass is not enough to for the model to get good performance. We use the following algorithm to train the perceptron:
```python
for n in range(number_of_epochs):
    for sample in samples:
        # Calculate output of the model
        weighted_sum = perceptron.weights*sample.inputs + sample.bias # sum = W^T * x + b
        perceptron_output = heaviside_step_function(weighted_sum)

        # Update weights of the model
        error = sample.target_output - perceptron_output
        for idx in perceptron.num_inputs:
            weight[i] += learning_rate * error * sample[i]
```


### Linear Separability

You can use perceptron when the dataset is **linearly separable**. This is the case if there exists one line that separates the categories in such a way that all the occurences of category A are on one side, and every occurence of category B is on the other. As we've seen before, the dataset used was not entirely linearly separable, no matter the line you drew, there were alwasy a few on the wrong side of the line. If we would generate more examples, the problem would worsen. However, we can take back the idea that was presented in the beginning: encoding the data.

Instead of using the amount and the time, we can encode these inputs as follows:

| Expression | Encoding |
| --- | --- |
| The amount is above 10k | $x_{amount} = 1$ |
| The amount is below 10k | $x_{amount} = 0$ |
| The time is between 12 and 6 AM | $x_{time} = 1$ |
| The time is not between 12 and 6 AM | $x_{time} = 0$ |

And the problem is reduced to the initial, simple exampl of whether to walk or not. Consider Figure 5, it shows the new plot of all the datapoints. Clearly, they are linearly separable.

[FIGURE 5 HERE]

### Perceptron as logical gates

Another approach of looking at a perceptron is by considering it as logical gates. Suppose all of the following:

$$
i = 2
b = -1
w_1 = w_2 = 1
x_1 = 0 or 1
x_2 = 0 or 1
$$

We then get the following truth table:
| x_1 | x_2 | y |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |
This is the truth table of an AND gate.

Similarly, we can construct a NAND gate:
$$
i = 2
b = 1
w_1 = w_2 = -1
x_1 = 0 or 1
x_2 = 0 or 1
$$

The truth table:
| x_1 | x_2 | y |
| --- | --- | --- |
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |
Now this exciting, because NAND is one of the universal logic gates, meaning that every other logical gate can be created by combining only NAND gates.

And this is where we find a very interesting observation. Suppose that we want to make a perceptron with the following truth table:
| x_1 | x_2 | y |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |
This is an exlusive OR gate, or XOR gate. Furthermore, this is not a linearly separable problem. Figure 6 shows this, no matter how you try, you cannot draw a line that will cleanly separate the dataset.

[FIGURE 6 HERE]

We know that by connecting multiple NAND gates we can create a XOR gate, which we know is a linearly separable problem. Now, would it not be amazing that by combining multiple perceptrons we would be able to solve these previously unsolvable problems (HINT: Neural Networks).

## C Code

> NOTE: This section will cover the implementation done in C. In case you're new to perceptrons, it is strongly recommended to read the previous sections before diving into the code. This section will assume some C knowledge. All the concepts implemented here can be replicated in your programming language of choice.

We first define a struct to group all the properties of the perceptron:
```c
/**
* @struct Perceptron
* @brief Represents a single perceptron
*/
typedef struct Perceptron {
    float* weights; // Array of input weights
    float bias; // Bias term. This is bias term that gets added to your perceptron, independent of the weights.
    float output; // The calculated output of the perceptron. This is a weighted sum using the weights and the inputs.
    int num_inputs; // The number of input connections. This is the same number of weights.
} Perceptron;
```

We define some more structs that will be used later on:
 
We define a dataset and sample where each sample has multiple inputs and one output. A dataset consists of multiple samples.
```c
/**
* @struct Sample
* @brief Represents a single sample to be fitted by the perceptron
*/
typedef struct Sample {
    float* inputs; // Array of inputs for this sample.
    int output; // The traget label for this sample.
} Sample;

/**
* @struct Dataset
* @brief Represents the dataset on which to fit the perceptron on. Consists of Samples
*/
typedef struct Dataset {
    Sample* samples; // Array of samples in this dataset.
    int num_samples; // The number of samples in this dataset.
} Dataset;
```

We define a struct that keeps the training parameters:
```c
/**
* @struct TrainingParameters
* @brief Combines all the possible training parameters in a single struct
*/
typedef struct TrainingParameters {
    float learning_rate; // The learning rate to be used in each training step.
    int num_training_steps; // The number of training steps to perform.
} TrainingParameters;
```
The number of training steps are the number of epochs.


To initialize the perceptron we have the following code:
```c
/**
* @brief Creates and initializes a new perceptron
*
* @param num_inputs Number of inputs for this perceptron
* @return Perceptron* Pointer to created perceptron, NULL if allocation fails
*/
Perceptron* create_perceptron(int num_inputs) {
    if (num_inputs <= 0) {
        fprintf(stderr, "Invalid parameters for neuron creation.\n");
        return NULL;
    }

    Perceptron* perceptron = malloc(sizeof(Perceptron));
    if (perceptron == NULL) {
        fprintf(stderr, "Memory allocation failed for neuron creation.\n");
        return NULL;
    }

    perceptron->weights = malloc(num_inputs * sizeof(float));
    if (perceptron->weights == NULL) {
        fprintf(stderr, "Memory allocation failed for weights.\n");
        free(perceptron);
        return NULL;
    }

    for (int w = 0; w < num_inputs; w++) {
        perceptron->weights[w] = (float)rand() / RAND_MAX - 0.5f;
    }

    perceptron->num_inputs = num_inputs;
    perceptron->bias = (float)rand() / RAND_MAX - 0.5f;

    return perceptron;
}
```
We randomly initialize the weights and bais with a value between -0.5 and 0.5. Other options are valid as well.

Next, we define the Heaviside step function which returns 1 if the given value is bigger or equal to 0, otherwise it returns 0:
```c
/**
 * @brief Applies the Heaviside step activation function
 *
 * @param value Input value to apply the function on
 * @return int 1 if value > 0, 0 otherwise
 */
int heaviside_step_function(float value) {
    return value >= 0 ? 1 : 0;
}
```

All the systems are now in place to start calculating the output of the perceptron:
```c
/**
 * @brief Calculates perceptron output for given inputs
 *
 * @param perceptron Pointer to perceptron
 * @param inputs Array of input values
 * @return float Perceptron output (0 or 1)
 */
float calculate_output(Perceptron* perceptron, float* inputs) {
    if (perceptron == NULL || inputs == NULL) return 0;

    float weighted_sum = perceptron->bias;
    for (int i = 0; i < perceptron->num_inputs; i++) {
        weighted_sum += perceptron->weights[i] * inputs[i];
    }

    return heaviside_step_function(weighted_sum);
}
```
As can be seen, it first calculates the weighted sum of the inputs and after takes the heaviside step function.

The final building block that is required is a way to update the weights:
```c
/**
 * @brief Updates perceptron weights and bias based on error
 *
 * @param perceptron Pointer to perceptron
 * @param inputs Array of input values
 * @param target_output Expected output
 * @param learning_rate Learning rate for weight updates
 */
void update_weights(Perceptron* perceptron, float* inputs, int target_output, float learning_rate) {
    if (perceptron == NULL || inputs == NULL) return;

    int error = target_output - perceptron->output;
    for (int i = 0; i < perceptron->num_inputs; i++) {
        perceptron->weights[i] += learning_rate * error * inputs[i];
    }

    perceptron->bias += learning_rate * error;
}
```

And finally, we combine the previous two functions into a method to train the perceptron:
```c
/**
 * @brief Trains perceptron on given dataset
 *
 * @param perceptron Pointer to perceptron
 * @param dataset Training dataset
 * @param training_parameters Training parameters
 */
void train(Perceptron* perceptron, Dataset* dataset, TrainingParameters* training_parameters) {
    if (perceptron == NULL || dataset == NULL || training_parameters == NULL) return;
    if (training_parameters->learning_rate <= 0.0f || training_parameters->learning_rate > 1.0f) return;

    for (int n = 0; n < training_parameters->num_training_steps; n++) {
        printf("Evaluating perceptron at step: %d", n);
        evaluate_perceptron(perceptron, dataset);
        for (int s = 0; s < dataset->num_samples; s++) {
            Sample* sample = &dataset->samples[s];
            perceptron->output = calculate_output(perceptron, sample->inputs);
            update_weights(perceptron, sample->inputs, sample->output, training_parameters->learning_rate);
        }
    }
}
```


