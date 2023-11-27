# deep-learning-challenge

## Background 
With my expertise in machine learning and neural networks, I was tasked by the nonprofit foundation Alphabet Soup to develop a tool that could determine the potential success of funding applicants' ventures. Using the provided dataset of over 34,000 previously funded organizations, obtained from Alphabet Soup's business team, I delved into columns capturing various organization metadata. 
These included identification markers like **EIN** and **NAME**, alongside critical factors such as **APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT**, and **IS_SUCCESSFUL** – which indicated the effectiveness of the funding allocation. My role was to harness this dataset's features to craft a binary classifier capable of predicting an applicant's success if funded by Alphabet Soup.

## Step 1: Preprocess the Data

In Step 1, I began by preprocessing the dataset, leveraging my knowledge of Pandas and scikit-learn’s StandardScaler(). The aim was to prepare the dataset for the subsequent steps, particularly for compiling, training, and evaluating the neural network model in Step 2.

To kick off the process, I uploaded the starter file to Google Colab and followed the guidelines outlined in the Challenge files. First, I read the charity_data.csv into a Pandas DataFrame, identifying the target and feature variables for the model while dropping the EIN and NAME columns. I then assessed the number of unique values in each column, investigating those with over 10 unique values to determine the data points for each unique value.

For columns with an abundance of unique values, I established a cutoff point based on the number of data points per unique value. This facilitated the grouping of "rare" categorical variables into a new value, termed "Other," to streamline the dataset. Using pd.get_dummies(), I encoded the categorical variables and proceeded to split the preprocessed data into features (X) and target (y) arrays. Employing the train_test_split function, I segregated the data into training and testing datasets.

Lastly, I standardized the training and testing features datasets by employing a StandardScaler instance. Initially, I fitted it to the training data and subsequently applied the transform function to ensure consistency and uniformity in the dataset's scaling.

## Step 2: Compile, Train, and Evaluate the Model
Using my knowledge of TensorFlow, I designed a neural network, crafting a deep learning model for binary classification to predict the success of Alphabet Soup-funded organizations based on the dataset's features. The initial step involved determining the number of inputs, which dictated the configuration of neurons and layers within the model. Once this was established, I proceeded to compile, train, and evaluate the binary classification model, focusing on calculating both its loss and accuracy.

Within the Google Colab file where I completed the preprocessing steps in Step 1, I began by constructing a neural network model using TensorFlow and Keras, setting the number of input features and determining the nodes for each layer. I created the first hidden layer, selecting an appropriate activation function. If necessary, I added a second hidden layer, also with a suitable activation function, and culminated the model with an output layer featuring an appropriate activation function. Checking the structure of the model ensured its coherence and functionality.

Next, I compiled and trained the model, incorporating a callback to save the model's weights every five epochs for consistency and potential future use. Evaluating the model using test data provided insights into its performance, revealing both the loss and accuracy metrics. Finally, I saved and exported the results to an HDF5 file, naming it AlphabetSoupCharity.h5 for reference and documentation purposes.

## Step 3: Optimize the Model
In Step 3, I set out to optimize my model using TensorFlow, aiming for a predictive accuracy surpassing 75%. Leveraging various methods, I sought to refine the model's performance by adjusting input data and exploring several strategies:

I began by creating a new Google Colab file, AlphabetSoupCharity_Optimization.ipynb, and importing necessary dependencies. Importing and reading the charity_data.csv into a Pandas DataFrame allowed me to preprocess the dataset as previously done in Step 1, ensuring adjustments aligned with optimization modifications.

With the goal of achieving higher than 75% accuracy, I tailored the neural network model, implementing adjustments based on optimization strategies. This involved potential modifications such as dropping or adding columns, creating additional bins for rare occurrences, altering the neuron count or layers, experimenting with different activation functions, and adjusting epoch numbers in the training regimen.

Throughout this optimization process, I aimed to iteratively refine the model's architecture and parameters. Once satisfied with the adjustments, I saved and exported the optimized results to an HDF5 file, naming it AlphabetSoupCharity_Optimization.h5 for record-keeping and analysis purposes.

## Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.
- Alphabet Soup, a nonprofit foundation, aims to optimize its funding allocation by leveraging machine learning and neural networks. Armed with a dataset encompassing 34,000+ organizations that have previously received funding, the task is to craft a binary classifier. This tool will analyze various features within the dataset, including identification markers like EIN and NAME, alongside critical organizational aspects such as APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION type, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, and IS_SUCCESSFUL. The objective? To predict the likelihood of success for potential ventures if funded by Alphabet Soup, thereby enhancing the foundation's decision-making process for future funding initiatives.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing
- What variable(s) are the target(s) for your model?
  - The IS_SUCCESSFUL variable is the target for my model which tells us if the company’s past funding was successful. 
- What variable(s) are the features for your model?
  - IS_SUCCESSFUL
- What variable(s) should be removed from the input data because they are neither targets nor features?
  - The variables that should be removed from the input data are EIN and Name because they are neither targets nor features. 

Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?
  - 80 neurons, 3 layers, and a ReLU activation function were selected for the neural network model because the number of features were dictated the number of hidden nodes as shown below:
    ![image](https://github.com/ciincing/deep-learning-challenge/assets/130705911/b2cd38f3-468a-4bfe-a7ee-2f4424fbfc14)
- Were you able to achieve the target model performance?
  - No, I was not able to achieve the target model performance of 75%. My attempt only garnered a 72.4% accuracy, which is still far from the desired goal.
    ![image](https://github.com/ciincing/deep-learning-challenge/assets/130705911/fb84c68e-ddda-4e91-8473-7750453bc084)
- What steps did you take in your attempts to increase model performance?
  - To make the model better, I increased the neurons in a part of the network. This helps it understand more complex patterns in the data, like how things are connected. Also, I made it learn more by giving it more chances to adjust its settings through more "epochs," or learning cycles. This helps it get better at predicting things accurately. But I had to be careful not to do this too much, because too many cycles could make the model too focused on the data it already knows and not good at predicting new stuff. So, I balanced it to keep things working well.
    ![image](https://github.com/ciincing/deep-learning-challenge/assets/130705911/3add9684-9c45-4295-b9e9-8eddae091bc5)

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
- Alphabet Soup utilized machine learning and neural networks on a dataset of 34,000+ funded organizations to refine funding allocation. Their aim was to create a model predicting venture success if funded by Alphabet Soup, using the IS_SUCCESSFUL variable as the target, signifying past funding success. The model consisted of 80 neurons across 3 layers with ReLU activation but fell short of the 75% accuracy target, reaching 72.4%. There's a need to fine-tune without overfitting, highlighting the importance of balance in adjustments. The best model variation achieved a 72.7% accuracy using different neuron counts and layer configurations. Considering the random forest classifier's resilience to outliers, it's suggested for the next steps. To improve, reducing epochs between 20-50 is recommended. 
