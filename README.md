# IBM Watson Developer Certification Study Guide
A Study Guide for Exam C7020-230 - IBM Watson V3 Application Development

The certification exam covers all of the aspects of building an application that uses Watson services. This includes a basic understanding of cognitive technologies, as well as a practical knowledge of the core APIs. Get up to speed with these resources:
- [Cognitive Computing Primer](http://ibm.co/2k5PAxf)
- [Watson Developer Cloud API Documentation](http://ibm.co/2jEaYwD)

The aim of this doc is to provide a more consolidated view of the required reading and study that is outlined in the [IBM Watson Professional Certification Program Study Guide Series](http://ibm.co/2iYtyP9). 

[Watson is accessed through IBM Bluemix](http://ibm.co/2jdqk8s)
### [Check out and play with Watson services on Bluemix](http://bit.ly/2jtpOUB)
[![IBM Bluemix Watson](http://watson.codes/IBM_Watson_Logo_300.gif)](http://bit.ly/2jtpOUB)

## High-level Exam Objectives

- [Section 1 - Fundamentals of Cognitive Computing](#section-1---fundamentals-of-cognitive-computing)
 - [1.1 Define the main characteristics of a cognitive system.](#11-define-the-main-characteristics-of-a-cognitive-system)
 - [1.2 Explain neural nets.](#12-explain-neural-nets)
 - [1.3 Explain machine learning technologies (supervised, unsupervised, reinforcement learning approaches).](#13-explain-machine-learning-technologies)
 - [1.4 Define a common set of use cases for cognitive systems.](#14-define-a-common-set-of-use-cases-for-cognitive-systems)
 - [1.5 Define Precision, Recall, and Accuracy.](#15-define-precision-recall-and-accuracy)
 - [1.6 Explain the importance of separating training, validation and test data.](#16-explain-the-importance-of-separating-training-validation-and-test-data)
 - [1.7 Measure accuracy of service.](#17-measure-accuracy-of-service)
 - [1.8 Perform Domain Adaption using Watson Knowledge Studio (WKS).](#18-perform-domain-adaption-using-watson-knowledge-studio-WKS)
 - [1.9 Define Intents and Classes.](#19-define-intents-and-classes)
 - [1.10 Explain difference between ground truth and corpus.](#110-explain-difference-between-ground-truth-and-corpus)
 - [1.11 Define the difference between the user question and the user intent.](#111-define-the-difference-between-the-user-question-and-the-user-intent)

- [Section 2 - Use Cases of Cognitive Services](#section-2---use-cases-of-cognitive-services)
 - [2.1 Select appropriate combination of cognitive technologies based on use-case and data format.](#21-select-appropriate-combination-of-cognitive-technologies-based-on-use-case-and-data-format)
 - [2.2 Explain the uses of the Watson services in the Application Starter Kits.](#22-explain-the-uses-of-the-watson-services-in-the-application-starter-kits)
 - [2.3 Describe the Watson Conversational Agent.](#22-explain-the-uses-of-the-watson-services-in-the-application-starter-kits)
 - 2.4 Explain use cases for integrating external systems (such as Twitter, Weather API).

- Section 3 – Fundamentals of IBM Watson Developer Cloud
 - 3.1 Distinguish cognitive services on WDC for which training is required or not.
 - 3.2 Provide examples of text classification using the NLC.
 - 3.3 Explain the Watson SDKs available as part of the services on Watson Developer Cloud.
 - 3.4 Explain the Watson REST APIs available as part of the services on Watson Developer Cloud.
 - 3.5 Explain and configure Natural Language Classification.
 - 3.6 Explain and configure Visual recognition.
 - 3.7 Explain how Personality Insights service works.
 - 3.8 Explain how Tone Analyzer service works.
 - 3.9 Explain and execute Alchemy Language services.
 - 3.10 Explain and configure Retrieve and Rank service.

- Section 4 - Developing Cognitive applications using Watson Developer Cloud Services
 
 - 4.1 Call a Watson API to analyze content.
 - 4.2 Describe the tasks required to implement the Conversational Agent / Digital Bot.
 - 4.3 Transform service outputs for consumption by other services.
 - 4.4 Define common design patterns for composing multiple Watson services together (across APIs).
 - 4.5 Design and execute a use case driven service choreography (within an API).
 - 4.6 Deploy a web application to IBM Bluemix.

- Section 5 - Administration & DevOps for applications using IBM Watson Developer Cloud Services

 - 5.1 Describe the process of obtaining credentials for Watson services.
 - 5.2 Monitor resource utilization of applications using IBM Watson services.
 - 5.3 Monitoring application performance on IBM Bluemix.
 - 5.4 Examine application logs provided on IBM Bluemix.


## Section 1 - Fundamentals of Cognitive Computing
### 1.1. Define the main characteristics of a cognitive system.

- Cognitive systems understand, reason and learn 
 - Must understand structured and unstructured data 
 - Must reason by prioritizing recommendations and ability to form hypothesis 
 - Learns iteratively by repeated training as it build smarter patterns 
- Cognitive systems are here to augment human knowledge not replace it 
- Cognitive systems employ machine learning technologies 
 - Supervised learning versus unsupervised learning 
- Cognitive systems use natural language processing 

### 1.2 Explain neural nets.

https://github.com/cazala/synaptic/wiki/Neural-Networks-101

- Neural Nets mimic how neurons in the brain communicate. 
- Neural networks are models of biological neural structures. 

Neurons are the basic unit of a neural network. In nature, neurons have a number of dendrites (inputs), a cell nucleus (processor) and an axon (output). When the neuron activates, it accumulates all its incoming inputs, and if it goes over a certain threshold it fires a signal thru the axon.. sort of. The important thing about neurons is that they can learn.

Artificial neurons look more like this:

![Artificial neurons](https://camo.githubusercontent.com/8b87e593fb9382c16a81cc059d994adec259a1c4/687474703a2f2f692e696d6775722e636f6d2f643654374b39332e706e67)

Video:
[![Neural Networks Demystified - Part 1: Data and Architecture](https://i.ytimg.com/vi/bxe2T-V8XRs/maxresdefault.jpg)](https://www.youtube.com/watch?v=bxe2T-V8XRs?v=VID)
[Neural Networks Demystified - Part 1: Data and Architecture](https://www.youtube.com/watch?v=bxe2T-V8XRs?v=VID)

So how does a Neural Network learn?
A neural network learns by training. The algorithm used to do this is called backpropagation. After giving the network an input, it will produce an output, the next step is to teach the network what should have been the correct output for that input (the ideal output). The network will take this ideal output and start adjusting the weights to produce a more accurate output next time, starting from the output layer and going backwards until reaching the input layer. So next time we show that same input to the network, it's going to give an output closer to that ideal one that we trained it to output. This process is repeated for many iterations until we consider the error between the ideal output and the one output by the network to be small enough.

#### 1.2.1.1. Explain the role of synapse and neuron

- A nueron operates by recieving signals from other nuerons through connections called synapses. 

#### 1.2.1.2. Understand weights and bias

- For each nueron input there is a weight (the weight of that specific connection).
- When a artifical neuron activates if computes its state by adding all the incoming inputs multiplied by it's corresponding connection weight. 
- But Neurons always have one extra input, the bias which is always 1 and has it's own connection weight. THis makes sure that even when all other inputs are none there's going to be activation in the nueron. 

![Weights and Bias](https://qph.ec.quoracdn.net/main-qimg-31d260a826ec73fce99ae098be5a7351)

#### 1.2.1.3. List various approaches to neural nets

[Types of artificial neural networks](https://en.wikipedia.org/wiki/Types_of_artificial_neural_networks)

#### 1.2.1.4. Explain forward and backward propagation

##### Feed Forward Propagation 

A feedforward neural network is an artificial neural network wherein connections between the units do not form a cycle. As such, it is different from recurrent neural networks.
In this network, the information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network.

Video:
[![Neural Networks Demystified Part 2: Forward Propagation](https://i.ytimg.com/vi/UJwK6jAStmg/maxresdefault.jpg)](https://www.youtube.com/watch?v=UJwK6jAStmg?v=VID)
[Neural Networks Demystified Part 2: Forward Propagation](https://www.youtube.com/watch?v=UJwK6jAStmg?v=VID)

##### Back Propagation 

Backpropagation, an abbreviation for "backward propagation of errors", is a common method of training artificial neural networks used in conjunction with an optimization method such as gradient descent. It calculates the gradient of a loss function with respect to all the weights in the network, so that the gradient is fed to the optimization method which in turn uses it to update the weights, in an attempt to minimize the loss function.

Backpropagation requires a known, desired output for each input value in order to calculate the loss function gradient – it is therefore usually considered to be a supervised learning method; nonetheless, it is also used in some unsupervised networks such as autoencoders. It is a generalization of the delta rule to multi-layered feedforward networks, made possible by using the chain rule to iteratively compute gradients for each layer. Backpropagation requires that the activation function used by the artificial neurons (or "nodes") be differentiable.

But how does the backpropagation work?
This algorithm adjusts the weights using Gradient Descent calculation. Let's say we make a graphic of the relationship between a certain weight and the error in the network's output:

![Artificial neurons](https://camo.githubusercontent.com/e6a0e02bd080acc585a622d2c03ca6e44a9e9adc/687474703a2f2f692e696d6775722e636f6d2f36565a6542706e2e706e67)

This algorithm calculates the gradient, also called the instant slope (the arrow in the image), of the actual value of the weight, and it moves it in the direction that will lead to a lower error (red dot in the image). This process is repeated for every weight in the network.

The goal of any supervised learning algorithm is to find a function that best maps a set of inputs to its correct output. An example would be a classification task, where the input is an image of an animal, and the correct output would be the name of the animal.
The goal and motivation for developing the backpropagation algorithm was to find a way to train a multi-layered neural network such that it can learn the appropriate internal representations to allow it to learn any arbitrary mapping of input to output.

Video:
[![Neural Networks Demystified - Part 4: Backpropagation](https://i.ytimg.com/vi/GlcnxUlrtek/maxresdefault.jpg)](https://www.youtube.com/watch?v=GlcnxUlrtek?v=VID)
[Neural Networks Demystified - Part 4: Backpropagation](https://www.youtube.com/watch?v=GlcnxUlrtek?v=VID)

##### 1.2.1.5 Gradient Descent 

Video:
[![Neural Networks Demystified - Part 3: Gradient Descent](https://i.ytimg.com/vi/5u0jaA3qAGk/maxresdefault.jpg)](https://www.youtube.com/watch?v=5u0jaA3qAGk?v=VID)
[Neural Networks Demystified - Part 3: Gradient Descent](https://www.youtube.com/watch?v=5u0jaA3qAGk?v=VID)

### 1.3 Explain machine learning technologies 
(supervised, unsupervised, reinforcement learning approaches).

##### 1.3.1. Explain the connection between Machine learning and Cognitive systems
Reference: [Computing, cognition and the future of knowing](http://www.research.ibm.com/software/IBMResearch/multimedia/Computing_Cognition_WhitePaper.pdf)

Machine learning is a branch of the larger discipline of Artificial Intelligence, which involves the design and construction of computer applications or systems that are able to learn based on their data inputs and/or outputs. The discipline of machine learning also incorporates other data analysis disciplines, ranging from predictive analytics and data mining to pattern recognition. And a variety of specific algorithms are used for this purpose, frequently organized in taxonomies, these algorithms can be used depending on the type of input required. 

Many products and services that we use every day from search-engine advertising applications to facial recognition on social media sites to “smart” cars, phones and electric grids are beginnin to demonstrate aspects of Artificial Intelligence. Most consist of purpose-built, narrowly focused applications, specific to a particular service. They use a few of the core capabilities of cognitive
computing. Some use text mining. Others use image recognition with machine learning. Most are limited to the application for which they were conceived. 

Cognitive systems, in contrast, combine five core capabilities:
- 1. They create deeper human engagement.
- 2. They scale and elevate expertise.
- 3. They infuse products and services with cognition.
- 4. They enable cognitive processes and operations.
- 5. They enhance exploration and discovery.

Large-scale machine learning is the process by which cognitive systems improve with training and use.

Cognitive computing is not a single discipline of computer science. It is the combination of multiple academic fields, from hardware architecture to algorithmic strategy to process design to industry expertise.

Many of these capabilities require specialized infrastructure that leverages high-performance computing, specialized hardware architectures and even new computing paradigms. But these technologies must be developed in concert, with hardware, software, cloud platforms and applications that are built expressly to work together in support of cognitive solutions.

##### 1.3.2. Describe some of the main machine learning concepts:

- [A Tour of Machine Learning Algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
- [List of machine learning concepts](https://en.wikipedia.org/wiki/List_of_machine_learning_concepts)
- [Supervised learning, unsupervised learning and reinforcement learning: Workflow basics](http://stats.stackexchange.com/questions/144154/supervised-learning-unsupervised-learning-and-reinforcement-learning-workflow)

##### 1.3.2.1. Supervised learning:

- We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and output. 

##### 1.3.2.1.1.Classification

- In a classification problem we are trying to predict results in a discrete output. In other words we are trying to map input variables into categories. 

##### 1.3.2.1.2.Regression/Prediction

- In a regression problem we are trying to predict results with a continous output meaning that we are trying to map input variables to some continous function. 

##### 1.3.2.1.3.Semi-supervised learning

- Semi-supervised learning are tasks and techniques that also make use of unlabeled data for training – typically a small amount of labeled data with a large amount of unlabeled data.

##### 1.3.2.2. Unsupervised learning:

- Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. Input data is not labeled and does not have a known result.

##### 1.3.2.2.1.Artificial neural network

- An Artificial Neural Network (ANN) is an information processing paradigm that is inspired by the way biological nervous systems, such as the brain, process information. 

##### 1.3.2.2.2.Association rule learning

- Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness.

##### 1.3.2.2.3.Hierarchical clustering

- Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:

 - Agglomerative: This is a "bottom up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
 - Divisive: This is a "top down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

##### 1.3.2.2.4.Cluster analysis

- Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters). 

##### 1.3.2.2.5.Outlier Detection

- The local outlier factor is based on a concept of a local density, where locality is given by {\displaystyle k} k nearest neighbors, whose distance is used to estimate the density. By comparing the local density of an object to the local densities of its neighbors, one can identify regions of similar density, and points that have a substantially lower density than their neighbors. These are considered to be outliers.

##### 1.3.2.3. Reinforcement learning

- These algorithms choose an action, based on each data point and later learn how good the decision was. Over time, the algorithm changes its strategy to learn better and achieve the best reward. Thus, reinforcement learning is particularly well-suited to problems which include a long-term versus short-term reward trade-off. 

### 1.4. Define a common set of use cases for cognitive systems.

Customer Call Centers
- Agent Assist: Q&A
- Problem Solved: Provides a natural language help system so call agents can rapidly retrieve answers to customer questions
 - Services used: Conversation, natural language answer retrieval, keyword extraction, and entity extraction
- Automation: Customer/Technical Support Tickets Routing
- Customer: Go Moment  
 - Problems Solved:
  - a) Detect the topic of a ticket and route to the appropriate department to handle it 
  - b) room service, maintenance, housekeeping
  - c) Escalate support tickets based on customersentiment
  - d) Route support requests to agents that already solved similar problems by detecting natural language similarities between new customer tickets and resolved ones.
- Services used: natural language (text) classification, keyword extraction, entity extraction, and sentiment/tone analysis

Physicians
- Expert Advisor:
 - Example: Watson Discovery Advisor
 - Problem Solved: Provides relevant medical suggestions and insights in natural language so physicians can more accurately diagnose patients.
 - Services used: Conversation + natural language answer retrieval, entity extraction 
 
Social Media
- Data Insights:
 - Partner: Ground Signal
 - Problem Solved: Extract useful insights from social media such as Instagram and Twitter by determining the content of photos and topics/sentiment of user posts.
 - Services used: keyword, entity, and sentiment/tone analysis
 
### 1.5. Define Precision, Recall, and Accuracy.

#### 1.5.1. [Precision:](#https://en.wikipedia.org/wiki/Precision_and_recall)
- Definition: Precision is the percentage of documents labelled as positive that are actually positive.
- Formula: True Positives/(True Positives + False Positives)

#### 1.5.2. [Recall:](#https://en.wikipedia.org/wiki/Precision_and_recall)
- Recall is the percent of documents labelled as positive were successfully retrieved.
- Formula: True Positives/(True Positives + False Negatives)

#### 1.5.3. Accuracy:
- Accuracy is the fraction of documents relevant to a query that were successfully retrieved.
- Formula: (True Positives + True Negatives)/Total Document Count

#### 1.5.4. Diagrams like this are often useful in capturing the True/False
Positive/Negatives described above:
[https://www.quora.com/What-is-the-best-way-to-understand-the-termsprecision-and-recall](#https://www.quora.com/What-is-the-best-way-to-understand-the-termsprecision-and-recall)

### 1.6. Explain the importance of separating training, validation and test data.

Normally to perform supervised learning you need two types of data sets:
 1. In one dataset (your "gold standard") you have the input data together with correct/expected output, This dataset is usually duly prepared either by humans or by collecting some data in semi-automated way. But it is important that you have the expected output for every data row here, because you need for supervised learning.
 2. The data you are going to apply your model to. In many cases this is the data where you are interested for the output of your model and thus you don't have any "expected" output here yet.

While performing machine learning you do the following:
 1. Training phase: you present your data from your "gold standard" and train your model, by pairing the input with expected output.
 2. Validation/Test phase: in order to estimate how well your model has been trained (that is dependent upon the size of your data, the value you would like to predict, input etc) and to estimate model properties (mean error for numeric predictors, classification errors for classifiers, recall and precision for IR-models etc.)
 3. Application phase: now you apply your freshly-developed model to the real-world data and get the results. Since you normally don't have any reference value in this type of data (otherwise, why would you need your model?), you can only speculate about the quality of your model output using the results of your validation phase.

The validation phase is often split into two parts:

 1. In the first part you just look at your models and select the best performing approach using the validation data (=validation)
 2. Then you estimate the accuracy of the selected approach (=test).

Hence the separation to 50/25/25.

In case if you don't need to choose an appropriate model from several rivaling approaches, you can just re-partition your set that you basically have only training set and test set, without performing the validation of your trained model. I personally partition them 70/30 then.

### 1.7. Measure accuracy of service.

The goal of the ML model is to learn patterns that generalize well for unseen data instead of just memorizing the data that it was shown during training. Once you have a model, it is important to check if your model is performing well on unseen examples that you have not used for training the model. To do this, you use the model to predict the answer on the evaluation dataset (held out data) and then compare the predicted target to the actual answer (ground truth).

A number of metrics are used in ML to measure the predictive accuracy of a model. The choice of accuracy metric depends on the ML task. It is important to review these metrics to decide if your model is performing well.

### 1.8. Perform Domain Adaption using Watson Knowledge Studio (WKS).

There is a great YouTube video series for Watson Knowledge Studio here:

Video:
[![Teach Watson with Watson Knowledge Studio](https://i.ytimg.com/vi/XBwpU97D5aE/maxresdefault.jpg)](https://www.youtube.com/watch?v=XBwpU97D5aE?v=VID)
[Teach Watson with Watson Knowledge Studio](https://www.youtube.com/watch?v=XBwpU97D5aE?v=VID)

IBM Watson Knowledge Studio is a cloud-based application that enables developers and domain experts to collaborate on the creation of custom annotator components that can be used to identify mentions and relations in unstructured text.
Watson Knowledge Studio is:
- Intuitive: Use a guided experience to teach Watson nuances of natural language without writing a single line of code
- Collaborative: SMEs work together to infuse domain knowledge in cognitive applications

Use Watson™ Knowledge Studio to create a machine-learning model that understands the linguistic nuances, meaning, and relationships specific to your industry.

To become a subject matter expert in a given industry or domain, Watson must be trained. You can facilitate the task of training Watson with Watson Knowledge Studio. It provides easy-to-use tools for annotating unstructured domain literature, and uses those annotations to create a custom machine-learning model that understands the language of the domain. The accuracy of the model improves through iterative testing, ultimately resulting in an algorithm that can learn from the patterns that it sees and recognize those patterns in large collections of new documents.

The following diagram illustrates how it works.
[![Watson™ Knowledge Studio](https://www.ibm.com/watson/developercloud/doc/wks/images/wks-ovw-anno.png)]

1. Based on a set of domain-specific source documents, the team creates a type system that defines entity types and relation types for the information of interest to the application that will use the model.
2. A group of two or more human annotators annotate a small set of source documents to label words that represent entity types, words that represent relation types between entity mentions, and to identify coreferences of entity types. Any inconsistencies in annotation are resolved, and one set of optimally annotated documents is built, which forms the ground truth.
3. The ground truth is used to train a model.
4. The trained model is used to find entities, relations, and coreferences in new, never-seen-before documents.

Deliver meaningful insights to users by deploying a trained model in other Watson cloud-based offerings and cognitive solutions.

Watson services integration

Share domain artifacts and models between IBM Watson Knowledge Studio and other Watson services.

Use Watson Knowledge Studio to perform the following tasks:
- Bootstrap annotation by using the AlchemyLanguage entity extraction service to automatically find and annotate entities in your documents. When human annotators begin to annotate the documents, they can see the annotations that were already made by the service and can review and add to them. See Pre-annotating documents with IBM AlchemyLanguage for details.
- Import industry-specific dictionaries that you downloaded from IBM® Bluemix® Analytics Exchange.
- Import analyzed documents that are in UIMA CAS XMI format. For example, you can import UIMA CAS XMI files that were exported from IBM Watson Explorer content analytics collections or IBM Watson Explorer Content Analytics Studio.
- Deploy a trained model to use with the AlchemyLanguage service.
- Export a trained model to use in IBM Watson Explorer.

### 1.9. Define Intents and Classes.

- The Natural Language Classifier service available via WDC, enables clustering or classification based on some measure of inherent similarity or distance given the input data. Such clustering is known as intents or classes.

- Where classes may include images, intent is a similar clustering for written utterances in unstructured natural language format.

### 1.10. Explain difference between ground truth and corpus.

- Ground truth is used in both supervised and unsupervised machine learning approaches, yet portray different values and formats. For example, in a typical supervised learning system, ground truth consisted of inputs (questions) and approved outputs (answers). With the aide of logistical regression and iterative training the system improves in accuracy.

- In unsupervised approach, such as NLC, the ground truth consists of a comma-separated csv or a JSON file that lists hundreds of sample utterances and a dozen or so intents (or classes) classifying those utterances.

### 1.11. Define the difference between the user question and the user intent.

To answer correctly, we need to understand the intent behind the question, in order to first classify it then take action on it (e.g., with a Dialog API)
- The user question is the verbatim question
- The user intent maps the user question to a known classification
- This is a form of classifying question based on search goals
- Intents are the superset of all actions your users may want your cognitive system to undertake. Put another way, questions are a subset of user intents. Questions usually end in "?", but sometimes we need to extract the user intent from the underlying context.
 - Common examples of user intents:
  - Automation: “Schedule a meeting with Sue at 5pm next Tuesday.”
  - Declarative: “I need to change my password.”
  - Imperative: “Show me the directions to my the nearest gas station.”
  
## Section 2 - Use Cases of Cognitive Services

### 2.1. Select appropriate combination of cognitive technologies based on use-case and data format.

#### 2.1.1. Agent-assist for email-based customer call center
 - Data: customer emails
 - Services: Q&A, Text classification, entity extraction and, keyword extraction
 - Watson-specific: NLC, R&R, Alchemy Language
 
#### 2.1.2. Agent-assist for phone-based customer call center
 - Data: customer voice recordings
 - Services: Q&A, Speech recognition, text-to-speech, text classification, entity extraction, keyword extraction
 - Watson-specific: NLC, R&R, Alchemy Language

#### 2.1.3. Expert advisor use case for physicians
 - Data: natural language intents
 - Services: Q&A, Text classification, entity extraction and keyword extraction
 - Watson-specific: NLC, R&R, Alchemy Language
 
#### 2.1.4. Data insights for Instagram images
 - Data: images
 - Services: Image classification and natural OCR
 - Watson-specific: Visual Insights

#### 2.1.5. Data insights for Twitter
 - Data: tweets
 - Services: Text classification, entity extraction, keyword extraction
 - Watson-specific: NLC and Alchemy Language
 
### 2.2. Explain the uses of the Watson services in the Application Starter Kits.
### 2.3. Describe the Watson Conversational Agent.

For section 2.2 and 2.3, we deep dive into the Watson services currently available and stated in the study guide. By understanding the services individually, it will help with knowing what services would work for different scenarios. 

[You can view the list of Watson Starter Kits here](https://www.ibm.com/watson/developercloud/starter-kits.html)

#### [Natural Language Classifier](https://www.ibm.com/watson/developercloud/doc/natural-language-classifier/index.html)

The IBM Watson™ Natural Language Classifier service uses machine learning algorithms to return the top matching predefined classes for short text inputs. The service interprets the intent behind text and returns a corresponding classification with associated confidence levels. The return value can then be used to trigger a corresponding action, such as redirecting the request or answering a question.

##### Intended Use

The Natural Language Classifier is tuned and tailored to short text (1000 characters or less) and can be trained to function in any domain or application.

- Tackle common questions from your users that are typically handled by a live agent.
- Classify SMS texts as personal, work, or promotional
- Classify tweets into a set of classes, such as events, news, or opinions.
- Based on the response from the service, an application can control the outcome to the user. For example, you can start another application, respond with an answer, begin a dialog, or any number of other possible outcomes.

Here are some other examples of how you might apply the Natural Language Classifier:
- Twitter, SMS, and other text messages
 - Classify tweets into a set of classes, such as events, news, or opinions.
 -  Analyze text messages into categories, such as Personal, Work, or Promotions.
- Sentiment analysis
 - Analyze text from social media or other sources and identify whether it relates positively or negatively to an offering or service.

##### You input
Text to a pre-trained model

##### Service output
Classes ordered by confidence

##### How to use the service
The process of creating and using the classifier:
[![Natural Language Classifier](https://www.ibm.com/watson/developercloud/doc/natural-language-classifier/images/classifier_process.png)]

##### CSV training data file format
Make sure that your CSV training data adheres to the following format requirements:
- The data must be UTF-8 encoded.
- Separate text values and each class value by a comma delimiter. Each record (row) is terminated by an end-of-line character, which is a special character or sequence of characters that indicate the end of a line.
- Each record must have one text value and at least one class value.
- Class values cannot include tabs or end-of-line characters.
- Text values cannot contain tabs or new lines without special handling. To preserve tabs or new lines, escape a tab with \t, and escape new lines with \r, \n or \r\n.
- For example, Example text\twith a tab is valid, but Example text    with a tab is not valid.
- Always enclose text or class values with double quotation marks in the training data when it includes the following characters:
- Commas ("Example text, with comma").
- Double quotation marks. In addition, quotation marks must be escaped with double quotation marks ("Example text with ""quotation""").

##### Size limitations
There are size limitations to the training data:
- The training data must have at least five records (rows) and no more than 15,000 records.
- The maximum total length of a text value is 1024 characters.

##### Supported languages

The classifier supports English (en), Arabic (ar), French (fr), German (de), Japanese (ja), Italian (it), Portuguese (pt), and Spanish (es). The language of the training data must match the language of the text that you intend to classify. Specify the language when you create the classifier.

##### Guidelines for good training
The following guidelines are not enforced by the API. However, the classifier tends to perform better when the training data adheres to them:
- Limit the length of input text to fewer than 60 words.
- Limit the number of classes to several hundred classes. Support for larger numbers of classes might be included in later versions of the service.
- When each text record has only one class, make sure that each class is matched with at least 5 - 10 records to provide enough training on that class.
- It can be difficult to decide whether to include multiple classes for a text. Two common reasons drive multiple classes:
- When the text is vague, identifying a single class is not always clear.
- When experts interpret the text in different ways, multiple classes support those interpretations.
- However, if many texts in your training data include multiple classes, or if some texts have more than three classes, you might need to refine the classes. For example, review whether the classes are hierarchical. If they are hierarchical, include the leaf node as the class.

[More detailed documentation for Natural Language Classifier](https://www.ibm.com/watson/developercloud/doc/natural-language-classifier/index.html)

#### AlchemyLanguage

AlchemyLanguage is a collection of APIs that offer text analysis through natural language processing. The AlchemyLanguage APIs can analyze text and help you to understand its sentiment, keywords, entities, high-level concepts and more.

##### Intended Use
Use one or all of the natural language processing APIs available through AlchemyLanguage to add high-level semantic information. Browse the documentation to learn more about each of AlchemyAPI's text analysis service functions:
- Entity Extraction
- Sentiment Analysis
- Emotion Analysis
- Keyword Extraction
- Concept Tagging
- Relation Extraction
- Taxonomy Classification
- Author Extraction
- Language Detection
- Text Extraction
- Microformats Parsing
- Feed Detection
- Linked Data Support

##### You input
Any publicly-accessible webpage or posted HTML/text document.

##### Service output
Extracted meta-data including, entities, sentiment, keywords, concepts, relations, authors, and more, returned in XML, JSON, and RDF formats

#### AlchemyData News

AlchemyData provides news and blog content enriched with natural language processing to allow for highly targeted search and trend analysis. Now you can query the world's news sources and blogs like a database.

AlchemyData News indexes 250k to 300k English language news and blog articles every day with historical search available for the past 60 days. You can query the News API directly with no need to acquire, enrich and store the data yourself - enabling you to go beyond simple keyword-based searches.

##### Intended Use

Highly targeted search, time series and counts for trend analysis and pattern mapping, and historical access to news and blog content.

##### You input

Build a query with natural language processing to search both the text in indexed content and the concepts that are associated with it.

##### Service output

News and blog content enriched with our full suite of NLP services.
Keywords, Entities, Concepts, Relations, Sentiment, Taxonomy

#### Tone Analyzer

The IBM Watson™ Tone Analyzer Service uses linguistic analysis to detect three types of tones from text: emotion, social tendencies, and language style. Emotions identified include things like anger, fear, joy, sadness, and disgust. Identified social tendencies include things from the Big Five personality traits used by some psychologists. These include openness, conscientiousness, extroversion, agreeableness, and emotional range. Identified language styles include confident, analytical, and tentative.

##### Intended Use

##### Common uses for the Tone Analyzer service include:

- Personal and business communications - Anyone could use the Tone Analyzer service to get feedback about their communications, which could improve the effectiveness of the messages and how they are received.
- Message resonance - optimize the tones in your communication to increase the impact on your audience
- Digital Virtual Agent for customer care - If a human client is interacting with an automated digital agent, and the client is agitated or angry, it is likely reflected in the choice of words they use to explain their problem. An automated agent could use the Tone Analyzer Service to detect those tones, and be programmed to respond appropriately to them.
- Self-branding - Bloggers and journalists could use the Tone Analyzer Service to get feedback on their tone and fine-tune their writing to reflect a specific personality or style.

##### You input
Any Text

##### Service output
JSON that provides a hierarchical representation of the analysis of the terms in the input message
