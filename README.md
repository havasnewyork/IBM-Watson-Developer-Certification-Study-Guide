# IBM-Watson-Professional-Certification-Study-Guide
A Study Guide for Exam C7020-230 - IBM Watson V3 Application Development
## High-level Exam Objectives

- [Section 1 - Fundamentals of Cognitive Computing](#section-1---fundamentals-of-cognitive-computing)
 - [1.1 Define the main characteristics of a cognitive system.](#11-define-the-main-characteristics-of-a-cognitive-system)
 - [1.2 Explain neural nets.](#12-explain-neural-nets)
 - 1.3 Explain machine learning technologies (supervised, unsupervised, reinforcement learning approaches).
 - 1.4 Define a common set of use cases for cognitive systems.
 - 1.5 Define Precision, Recall, and Accuracy.
 - 1.6 Explain the importance of separating training, validation and test data.
 - 1.7 Measure accuracy of service.
 - 1.8 Perform Domain Adaption using Watson Knowledge Studio (WKS).
 - 1.9 Define Intents and Classes. 
 - 1.10 Explain difference between ground truth and corpus.
 - 1.11 Define the difference between the user question and the user intent.

- Section 2 - Use Cases of Cognitive Services
 - 2.1 Select appropriate combination of cognitive technologies based on use-case and data format.
 - 2.2 Explain the uses of the Watson services in the Application Starter Kits.
 - 2.3 Describe the Watson Conversational Agent.
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
#### 1.1. Define the main characteristics of a cognitive system.

- Cognitive systems understand, reason and learn 
 - Must understand structured and unstructured data 
 - Must reason by prioritizing recommendations and ability to form hypothesis 
 - Learns iteratively by repeated training as it build smarter patterns 
- Cognitive systems are here to augment human knowledge not replace it 
- Cognitive systems employ machine learning technologies 
 - Supervised learning versus unsupervised learning 
- Cognitive systems use natural language processing 

#### 1.2 Explain neural nets.

https://github.com/cazala/synaptic/wiki/Neural-Networks-101

Neurons are the basic unit of a neural network. In nature, neurons have a number of dendrites (inputs), a cell nucleus (processor) and an axon (output). When the neuron activates, it accumulates all its incoming inputs, and if it goes over a certain threshold it fires a signal thru the axon.. sort of. The important thing about neurons is that they can learn.

Artificial neurons look more like this:

![GitHub Logo](https://camo.githubusercontent.com/8b87e593fb9382c16a81cc059d994adec259a1c4/687474703a2f2f692e696d6775722e636f6d2f643654374b39332e706e67)

[![Neural Networks Demystified - Part 1: Data and Architecture](https://i.ytimg.com/vi/bxe2T-V8XRs/maxresdefault.jpg)](https://www.youtube.com/watch?v=bxe2T-V8XRs?v=VID)

