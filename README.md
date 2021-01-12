# UKElectionPrediction
# **Abstract**

![](RackMultipart20210112-4-ahg56g_html_f16e5fb295dbf71c.gif)

Predicting UK Election Outcome using Political News Sentiment and Constituency Socioeconomic Conditions

by

Seth Odia

Supervised by

Dr. Gihan Mudalige

for the degree of

MSc Computer Science

Department of Computer Science

University of Warwick

September 2020

In this project I aim to be able to accurately predict the outcome of UK elections by predicting what party wins in the various Westminster Parliamentary constituencies in England. This will be done using data from news sites and blogs as well as socioeconomic conditions in each constituency around England and derive relationships between these ![](RackMultipart20210112-4-ahg56g_html_36e25db06c5e3e9b.jpg) data sources and election results with the use of machine learning. To do this, I am going to make use of various classification models such as AdaBoost, Random Forest, Decision Tree and Support Vector Machine.

**Keywords** - Classification, Machine Learning, Natural Language Processing, Sentiment Analysis, Correlation, Elections, Politics, Data Mining.

![](RackMultipart20210112-4-ahg56g_html_1d1d70b63fd8e21d.gif)

I would like to thank and express my deepest gratitude to Dr. Gihan Mudalige for supervising and supporting the development of this project.

**Acknowledgements**

# **Contents**

**Abstract i**

**Acknowledgements ii**

1. **Introduction 1** 1.1 Objectives 2 1.2 Changed Objectives 3 1.3 Resources &amp; Tools 3

1. **Literature Review 5** 2.1 Internet Effect on Elections 5 2.2 Web Scraping 5 - 2.2.1. Application Programming Interfaces 6 - 2.2.2 Screen Scraping 7 2.3. Election Prediction 7 2.4 Machine Learning for Election Prediction 8 2.5. NLP for Election Prediction 10

1. **Project Management 12** 3.1 Timeline 12 3.2 Risk Assessment 14 3.3 Coordination with Supervisor 16 3.4 Technical Decisions 16
2. **Methodology 17** 4.1 Political News Data 18 - 4.1.1 Conservative Leader News 19 - 4.1.2 Conservative Party News 21 - 4.1.3 Labour Leader News 21 - 4.1.4 Labour Party News 22 4.2 Sentiment Analysis 22 4.3 Constituency Socioeconomic Conditions Data 23 - 4.3.1 Proportion Owning House 24 - 4.3.2 Age Distribution 24 - 4.3.3Proportion 25 - 4.3.4 Percentage of Students Achieving 5A\*-C Grades at GCSE 25 - 4.3.5 Median House Price Born in the UK 25 - 4.3.6 Brexit Vote 25 - 4.3.7 Median Wage 25 4.4 Election Result Data 26 4.5 Preprocessing 26 - 4.5.1 Data Quality Assessment 26 - 4.5.2 Inconsistent Values 27 - 4.5.3 Feature Encoding 27 4.6 Training 28 4.7 Evaluation 28
3. **Implementation and Results**
4. **Testing**
5. **Conclusion**

# **Chapter
# 1**

# **Introduction**

The United Kingdom practices a parliamentary system of government which is a democratic form of government in which the party (or a coalition of parties) with the greatest representation in the parliament (legislature) forms the government, its leader becoming Prime Minister. Executive functions are exercised by members of the parliament appointed by the prime minister to the cabinet. The parties in the minority serve in opposition to the majority and have the duty to challenge it regularly.[3] The UK uses the First Past the Post electoral system to elect members of parliament to Westminster. In this system, during a General Election, 650 constituencies across the UK each hold separate contests. To become an MP, a candidate needs the largest number of votes in their area.[4] Which ever party wins the majority of the 650 constituencies forms the government with their party leader becoming the Prime Minister.

Predicting elections using traditional methods has proven to be problematic as issues arise from its accuracy to the amount of time and effort required to make these predictions. Also, the political views of people across the population have overtime become more diverse and non identical to someone else from a similar background. This can be seen as an effect of the internet age where people consume information from an abundance of diverse sources like social media sites such as Facebook and Twitter and online news and blog sites such as TMZ and Huffington Post.

Such platforms have an ability to influence peoples political views and beliefs and because there are so many of these platforms compared to a number of decades ago where there were only a number of newspaper companies and TV channels, it has led to the public having a more diverse view on political subjects and thus more diverse opinions during elections. This has made making predictions more difficult and inaccurate, and because of this, people such as electorates and candidates end up believing wrong predictions. For the election candidates, wrong predictions can affect their campaigns negatively as for instance, they end up spending more time campaigning where they believe they are less popular instead of the places where more campaigning will make a difference.

Traditional election prediction methods such as the exit polls which is a survey of thousands of voters just after they have cast their ballot. The exit poll is based on 144 constituencies in England, Scotland and Wales. The constituencies are chosen to be demographically representative of the country, balanced between rural and urban seats, and weighted slightly in favour of marginal areas. Although its accuracy has increased in recent years, it still has a long way to go and this can be seen in the 2015 election where the exit polls although predicting a Conservative win, did not predict a Conservative majority and in 2017, the first take of the exit poll correctly predicted the Conservatives would be the largest party, but stopped short of saying there would be a hung Parliament.[1]

Another traditional method are the opinion polls which provide a snapshot of people&#39;s views on a particular issue. During the 2019 election campaign YouGov published a poll based on the views of 100,000 voters, applying national trends to individual constituencies. Over the last few elections, 90% of polls conducted just before the election have been within four percentage points of the vote share for the major parties, but are not even as accurate as the exit polls.[2]

The aim of this project is to obtaincorrelation between election results and voter sentiment in regards to political news as well as socioeconomic conditions in each constituency around England. From this it it is then possible to ascertain which attributes were the most influential in this election and to predict voting patterns without having to make use of a poll. This project is motivated by the aspiration to further comprehend how media consumption and economic and socialattributesinfluence voting choices.

### **1.1 Objectives**

To achieve the overall aim of the project, it has been broken down in several tasks. These objectives, are reiterated below.

- Collect data of past elections with breakdown of election outcome.
- Scrape online data from news sites involving political parties and candidates.
- Clean data and form datasets with the obtained data
- Make use of Natural Language Processing on news data to derive voter sentiment on candidates and political parties.
- Preprocess dataset attributes to make data more presentable to the model in order to provide better predictions
- Test various models to find which works best for predicting election results in the Westminster Parliamentary constituencies in England.
- Fine tune model parameters to have as low bias as possible while also having as low variance as possible on training data.
- Split datasets into training and test data and train models with data.
- Test models with test data and measure accuracy using metrics such as AUC-ROC, F1 Score, Mean Squared Error.
- Evaluate which features affect election results the most.
- Visualize results using map of England showing output from each constituency

### **1.2 Changed Objectives**

Some of the objectives have been changed as the project have evolved some of these are :

- Predicting UK elections instead of Nigeria elections
- Using news data to get sentiment as an alternative of using twitter data
- Using social and economic constituency factors as attributes along with sentiment rather than just using sentiment alone.

### **1.3. Resources &amp; Tools**

This project relies on using the Python programming language. I chose to use Python because it is easy and flexible to use, versatile and has a robust collection of libraries that make machine learning tools easily available to use. To write this python code I used the Jupiter notebook and Colab applications.

I used various Python libraries to access tools to bring my project to life many of these being machine learning tools such as:

- Sklearn
- Matplotlib
- Pandas
- Numpy

To accumulate online news data, I made use of the BeautifulSoup library a tool which enables one to scrape text from websites. Then to process the text in order to draw out sentiment, the nltk library is used.

# **Chapter
# 2**

# **Literature Review**

This chapter aims to look at existing literature which is used to form a context for this project. Such literature are based on subjects surrounding the structure of the UK elections , techniques used to make predictions in election results including both traditional methods and more recent methods such as building and training machine learning models and the application of Natural Language Processing to politics.

### **2.1 Internet Effect on Elections**

In recent years, because of the increase in popularity of the internet, studies have been undergone to see how the internet has affected elections. Grofman(2014)talks about how terms such as e-democracy have morphed over the years from speculative and optimistic accounts of a future heightened direct citizen involvement in political decision-making and an increasingly withered state apparatus, to more prosaic investigations of party and governmental website content and micro level analyses of voters&#39; online activities.[5] In this book, he investigates the role of digital media and competitive advantage, campaigns and the effect of social media, online communication as way of fomenting nonviolent revolutions and the undeniable and important role of the internet on democracy around the world. He uses the 2010 national elections in Brazil as a research point due to the fact that an expandingsum total of Brazilian candidates had begun to use websites and social networking applications as an essential part of their overall campaign efforts and findings in his study suggest that certain internet sites can be better suited for political campaign purposes than others and that this most likely varies depending upon the political and electoral systems, as well as culture giving an example of how Twitter was a more suitable media campaign platform compared to Facebook and Orkut. He then ends with stating that these new media connections between candidates and voters can play a central role in shoring up voter support.

A study by Bimber (2003) shows that the internet facilitates highly selective forms of engagement, we see evidence that through use by campaigns, supporters of candidates more engaged even if the effect is not overwhelming in scale and even if it is primarily online. If at least some politically interested individuals can find a channel for their desire for activism, that would be an attractive development.[7]. This proves that the media has an effect on voter preferences and with the right media coverage during an election a candidate or party has a greater chance of winning an election.

With influential people in politics having some sort of control over the media, and being able to use the media to influence voters, there has been a rise in propaganda used to win support, MacLeod (2019) studies how propaganda is used to influence the public and lifting the veil over how the mass media operates to show how effective it has been with a focus on past events such as fake news, Cambridge Analytica, the Syrian Civil War and RussiaGate.[8]

### **2.2 Web Scraping**

[Web Scraping](http://en.wikipedia.org/wiki/Web_scraping) is a technique employed to extract large amounts of data from websites whereby the data is extracted and saved to a local file in your computer or to a database in table format.[12] The internet contains the largest pool of data that can be harnessed and with the use of web scraping tools it is possible to obtain this.Unstructured data is gathered from the web and then converted into more structured datasets perhaps in the from of a table. There are two forms of web scraping which will be discussed below

### **2.2.1 Application Programming Interfaces**

This form of web scraping entails using application programming interfaces, commonly referred to as APIs. This is where a website provides a set of structured HTTP requests that return JSON or XML files. Some of the most popular APIs for this are as follows[14]:

- Scraper Box API - This is a simple API that allows you to scrape data from online resources without experiencing blockades and returns the scraped data in HTML format.
- Scraper Crawler Extract API - This allows you to specify the URL of a web page and retrieve its contents fast and returns data as a JSON array.
- ScrapingBee API - This allows you to handle the various web scraping challenges so that you can harvest data without any worries and returns HTML formatted responses.

Dewi (2019) makes use of the Facebook and Twitter API to scrape data from social media in a way that the user is able to search information, combine and present it in a better way according to user preferences.[15]

### **2.2.2 Screen Scraping**

In this method, you extract data from source code of a website, with an HTML parser or regular expression matching. A prior knowledge of HTML and CSS is helpful while making use of this method to extract data. There are various HTML parser libraries that are compatible with specific programming languages.

- The Java Programming language has parser libraries such as &#39;Largato and Jerry&#39; and Jsoup.
- Python has parser libraries such as BeautifulSoup, [HTML Parser of The Standard Library](https://tomassetti.me/parsing-html/#standardhtml) and [Html5lib](https://tomassetti.me/parsing-html/#html5lib).
- JavaScript has parser libraries such as DOMParser, Cheerio and Parse5

### **2.3 Election Prediction**

A lot of research has gone into election prediction. There are traditional methods that have been around for a while such as opinion polls and exit polls and also newer methods such as using social media data along machine learning algorithms to make predictions. Research has been done using various elections as a sample and different social media sources such as the 2011 dutch senate election results with twitter[11] and the recent Irish General Election as a case study for investigating the potential to model political sentiment through mining of social media.

Fabio (2012) measures the predicting power of social media regressing one variable at a time using an ARIMA model which stands for &#39;Auto Regressive Integrated Moving Average&#39;, also known as Box–Jenkins model[9]. This model is fitted to [time series](https://en.wikipedia.org/wiki/Time_series) data either to better understand the data or to predict future points in the series.[10]. Using data from social media sites such as Twitter and Facebook, he predicts the 2010 UK election results accurately with a 0.19338 percentage point error with an average Conservative prediction equal to 34.298788 and 34.265892, Labour prediction equal to 29.75584 and 28.46665 and Liberal Democrats prediction equal to 23.18733 and 26.28944. Overall, the ARIMA model seems to bypass unrepresentative sample flaws and leads to forecasts that surpass in accuracy those of the more traditional and expensive polls.

### **2.4 Machine Learning for Election Prediction**

Machine learning is a tool that has been applied in various fields ranging from science and medical research for example using machine learning models to predict if someone has cancer, to financial research like producing stock market trends and of course politics is not left out when it comes to ways in which machine learning can be applied as new approaches for predicting various types of elections are being researched and implemented. Various types of datasets and machine learning algorithms are used to achieve these goals on different kinds of elections around the world. Having researched some of them I have discussed them below examining the datasets used, algorithms applied and results achieved.

León-Borges(2017) compares three different machine learning algorithms namely, grouping method K-means, expectation a convergence criteria, EM and methodology for classification LAMDA. To do this he makes use of the Weka and SALSA software to carry out this classification and he applies the algorithms on the local governor elections in 1998, 2004 and 2010 of the state of Quintana Roo, Mexico[16]. He then concluded that the LAMBDA algorithm used along with the SALSA software worked best as it gave the most accurate results.

Tsakalidis (2015) uses data extracted from Twitter and opinion polls to predict the 2014 European Union elections in three countries namely Germany, The Netherlands and Greece with the use of machine learning models such as linear regression, Gaussian process, and sequential minimal optimisation for regression, all implemented using Weka[17]. Gaussian process achieved the lowest Mean Absolute Error (1.31), followed by sequential minimal optimisation (1.35) and linear regression (1.42).

Ramteke (2016) tries to predict the winner of the 2016 American presidential elections between Donald Trump and Hilary Clinton using Twitter data and making use of two machine leaning algorithms named Multinomial Naive Bayes and Support Vector machines[18]. To predict the winner he uses machine learning to know if a tweet is positive or negative and then the candidate with the overall highest positive out of total tweet percentage is predicted as the winner.

Sharma (2016) utilised Dictionary Based, Naive Bayes and SVM algorithm to build their classifier and classified Twitter test data as positive, negative and neutral in order to predict general state elections of India in 2016. Tweets with the names of Indian political parties such as #BJP, #Congress, #NCP, #AAP were collected, and a total of 23,998 tweets relevant to these hashtags were collected. The results of the analysis for Naive Bayes and the SVM was the Bhartiya Janta Party as the winner and for the Dictionary Approach it was the Indian National Congress. SVM predicted a 78.4% chance that the BJP would win more elections in the general election due to the positive sentiment they received in tweets. As it turned out, BJP won 60 out of 126 constituencies in the 2016 general election, far more than any other political party as the next party (the Indian National Congress) only won 26 out of 126 constituencies[19].

### **2.5 NLP for Election Prediction**

NLP which stands for Natural Language Processing deals with the interaction between computers and humans using the natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human languages in a manner that is valuable[20]. One of the uses of NLP is Sentiment Analysis, this is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment Analysis can be used to classify emotion of people in a country regarding a political party or candidate which can then be used to predict voting patterns in an election and therefore predict the winner of the election.

Oyebode (2019) uses Sentiment Analysis to predict election outcome in the 2019 Nigeria Presidential elections. To achieve this, he uses VADER a widely used lexicon-based technique which is used to extract sentiment from a piece of text on a range of -1 to +1. The VADER lexicon consists of 7,517 opinion words and text emoticons that express either negative or positive sentiments. This lexicon is then applied on text data gotten from we scrapping Nairaland an indigenous social media platform targeted at Nigerians[21].

Tsakalidis (2015) also uses sentiment analysis for its European Union countries election prediction by using a sentiment lexicon to train a model and use it to differentiate between positive and negative tweets from Twitter datasets. This resulted in _14,060/19,357_ German, _13,838/18,993_ Dutch, and _13,582/18,356_ Greek positive/negative terms. To detect a tweet&#39;s sentiment, they used a naive sum-of-weights method on its keywords according to the respective lexicon, and assigned the majority class label (positive/negative) to it[17].

Sharma (2016) made use of SentiWordnet which contained synonyms and antonym of the respective words along with the score of that particular word. This was then used on the tweet data and keywords were compared and assigned polarity using a scoring scheme of +0.1 to +1.0 as positive, +0.1 to -1.0 as neutral and -0.1 to -1.0 as negative[19].

Ramteke (2016) makes use of manual labelling using Hash Tag Clustering and VADER the lexicon and rule-based sentiment analysis tool. First tweets are labelled based on their hashtag as positive or negative and then VADER is used to label the rest of the data. This data is then used to train a model with the ability to label more tweet data and measure its accuracy using the manual labelled and VADER labeled data. Multinomial Naive Bayes and Support Vector machines were used as machine learning algorithms to make models using the nltk and Scikit learn packages and the Scikit learn SVM model came out with the highest accuracy at 0.99 percent.

# **Chapter 3**

# **Project Management**

In this chapter, I aim to discuss and describe how the project has been managed and evolved since its inception from its original form at the start of this year in the form of a research proposal. The project was originally planned to apply machine learning techniques and algorithms to predict election outcome in Nigeria using social media. As the literature was reviewed and data was collected, the project&#39;s topic and focuschanged to UK election predictions due to the lack of data on previous Nigerian elections.

### **3.1 Timeline**

The project was split into multiple parts with each of the parts having its own time duration and deadline in order to ensure progression was ongoing. While these deadlines existed, they were not completely strict therefore allowing room for flexibility and therefore changes could be made to these deadlines to better suit my progress at that point and the sort of results being achieved at that moment. The project duration lasted for about 36 weeks which began in January 2020 and lasted up until 24 September 2020. My timeline was built around the four submissions spread out throughout the project and these were the Research Proposal due on the 20th of February 2020, Project Presentation due on the 24th of April 2020, Interim Report due on the 24th of June 2020 and finally the Dissertation Report due on the 10th of September 2020. These submissions helped keep me on track and be able to access my progress at each stage. Feedback from my supervisor for these submissions also was helpful in assessing my progress and knowing what was being doing well, what needed to improved upon, what features were not necessary and therefore could be removed and new features which would improve my project.

Due to the flexibility of the timeline, the initial timeline had multiple differences compared to the final timeline as objectives changed slightly and deadlines shifted.

Figure 3.1 Initial Project Timeline as Submitted in Research Proposal

Figure 3.2 Project Ti ![](RackMultipart20210112-4-ahg56g_html_92fb7cb2ca9006ae.png) ![](RackMultipart20210112-4-ahg56g_html_40e3cd036ff5c5cf.png) ![](RackMultipart20210112-4-ahg56g_html_b43621397f4755bd.png)meline as at Presentation date

Figure 3.3 Project Timeline as at Interim Report Submission

Through the period working on my project the timeline plan changed relatively as shown in the difference between Figure 3.1 all the way to Figure 3.3, this was due to the pandemic affecting my workflow as I was unable to make use of the university computers which had faster processing powers compared to my personal computer. Also, due to the fact I had coursework and exams while working on the project it was necessary for me at theseintervals to prioritise less on the project to concentrate on them. My health also played a role in adjusting my timeline as I unfortunately became ill twice and it became necessary to take a short break. While I started most stages on time, I did not necessarily end at eh expected time as different stages blended with each other along the way and I ended up working on multiple stages together for example, the preprocessing and building machine learning models stage. With all that I still ensured I met all submission deadlines on time roadblocks The first stage of my project after the initial research proposal which kicked off the project was to gather data. This took longer than I expected as finding available online text to provide sentiment was harder than I had initially though it would be. The solution to this was me scraping data of google news headlines to get the relevant data required for this project.

### **3.2 Risk Assessment**

There were three potential risks notified before embarking on my my project. These were:

**Underestimating Time to Complete Tasks**

Severity: Medium

Probability: Medium

Reason: A task may be more complicated to complete the initially expected. If the difference is sever this can greatly affect the project progress but a small difference will not greatly affect the project.

Strategy: This has been taken care of as agile methodology allows flexibility in the project to readjust. Also enough time is given in the end during dissertation writing stage that any task that is still not done can be done within that period.

**Illness Causing Delays**

Severity: High

Probability: Low

Reason: As the project is done individually, illness has the chance of stopping progression for extended periods of time.

Strategy: Illness that is not so serious can be handled if there are breaks in progress as project is agile and therefore flexible enough to accommodate a break caused by an illness.

**Delay due to unforeseen circumstances**

Severity: Medium

Probability: High

Reason: Due to project been done over a long period of time it is unable to predict everything happening to detail and know of any random event causing delay. The pandemic false under this as it was unforeseen and caused delay in executing my project

Strategy: This has been taken care of as agile methodology allows flexibility in the project to readjust. Also enough time is given in the end during dissertation writing stage that any task that is still not done can be done within that period.

### **3.3 Coordination with Supervisor**

Regular meetings were held with my supervisor throughout the duration of this project. These meetings were held mostly fortnightly with additional meetings being organised if there was an issue that needed to be discussed before the next official meeting date. Initial meetings were used to set a structure for the project knowing what will be feasible and the necessary tools I would need to successfully complete my project. In later these meetings, updates were given to my supervisor on my updates so far and what goals or objectives were in line to be achieved next,My supervisor also use this as an opportunity to give me feedback on what he thinks about my progress so far, letting me know if I am on the right path and makes recommendations on what I can include or remove from my project to make it better.

### **3.4 Technical Decisions**

For this project, the kanban methodology has been adopted as well as parts of the agile methodology. Kanban works by making a visual system for managing work as it moves through a process. Kanban visualizes both the process (the workflow) and the actual work passing through that process. The goal of Kanban is to identify potential bottlenecks in your process and fix them so work can flow through it cost- effectively at an optimal speed or throughput[22]. I have also adopt some elements of agile which will allow my project to be less rigid and be able to evolve over time as it emphasises flexibility, continuous improvement, and high quality results[23], by always accepting changes to the requirements in order to be able to achieve the aim of the project.

# **Chapter 4**

# **Methodology**

![](RackMultipart20210112-4-ahg56g_html_115173aafb660b0c.gif) ![](RackMultipart20210112-4-ahg56g_html_d609edff8f8e3c65.gif) ![](RackMultipart20210112-4-ahg56g_html_4739d8a83158decb.gif) ![](RackMultipart20210112-4-ahg56g_html_b714ad4ab738886c.gif) ![](RackMultipart20210112-4-ahg56g_html_b2e812369217370e.gif) ![](RackMultipart20210112-4-ahg56g_html_7902d4d968377a3a.gif) ![](RackMultipart20210112-4-ahg56g_html_7902d4d968377a3a.gif) ![](RackMultipart20210112-4-ahg56g_html_bb881356c96768b5.gif) ![](RackMultipart20210112-4-ahg56g_html_ad17b93b71f978f3.gif)
 ![](RackMultipart20210112-4-ahg56g_html_eabaabd6ba48450b.gif) ![](RackMultipart20210112-4-ahg56g_html_e2bfaa0655d60918.gif) ![](RackMultipart20210112-4-ahg56g_html_3c8534944571f9d7.gif) ![](RackMultipart20210112-4-ahg56g_html_fecf38d3c6a25a6c.gif)

Evaluate

**Model**

Election Result

Constituency

Sentiment Analysis

Political News

In this chapter, the methodology which will be used to achieve building accurate models to predict UK elections by predicting the winner in each constituency in England and then overall in the UK using the party with the highest sum of constituencies won. There are several components of the methodology used to achieve a functional model and these will be looked into in detail.. The following sections are required to achieve the creation of an efficient model for this project:

- Political News Data
- Sentiment Analysed Data
- Constituency Data
- Election Result Data
- Train model
- Test model
- Evaluation

Figure 4.1 Diagram of Training Model Structure

Figure 4.1 showcases the structure of the model in its training phase. The training dataset consists of political news, sentiment analysis, constituency data and election results which are the inputs inserted to the model which its behaviour will depend on a machine learning algorithm. Constituency and political news data fall under the input vector while election results is the corresponding output vector which is the target(label) of the input.

### **4.1 Political News Dataset**

For this dataset a compilation of news headlines and its summaries were collected. For each of the past elections being analysed (2019, 2017, 2015, 2010, 2005) top news data from [google.com](http://google.com/) was collected in the time periods leading up to the elections, these time periods were divided into 6 parts. The first being 5 months before the elections and the subsequent ones each a month closer to the election with the last one being the month of the election. In each month, the top 10 news stories were scraped. This was done using the requests library which is a simple HTTP library for Python which allows you to send HTTP requests extremely easily. without the need to manually add query strings to your URLs; and the BeautifulSoup library which is a Python library for pulling data out of HTML and XML files. A sample of the script used is shown below:

1 def \_\_init\_\_(self, term,start,end):

2 self.term = term

3 self.start = start

4 self.end = end

5 self.sentiment = 0

6 self.subjectivity = 0

7 self.url = &#39;[https://www.google.com/search?](https://www.google.com/search?)8tbs=cdr%3A1%2Ccd\_min%{1}%2Ccd\_max%{2}&amp;q={0} 9 &amp;source=lnms&amp;tbm=nws&#39;.format(self.term,self.start,self.end)

10

11

12 def run(self):

13 headers = {

14 &quot;User-Agent&quot;:

15 &quot;Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36

16 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36&quot;

17 }

18 response = requests.get(self.url, headers=headers)

19 soup = BeautifulSoup(response.text, &#39;html.parser&#39;)

20 headline\_results1 = soup.find\_all(&#39;div&#39;, class\_=&#39;JheGif nDgy9d&#39;)

21 headline\_results = soup.find\_all(&#39;div&#39;, class\_=&#39;Y3v8qd&#39;)

22 #open the spreadsheet we will write to

23 with open(&#39;%s%s.csv&#39; % (self.term,self.start[-4:]), &#39;a&#39;) as file:

24 w = csv.writer(file)

25for text1,text in zip(headline\_results1,headline\_results):

26 blob1 = TextBlob(text1.get\_text())

27blob = TextBlob(text.get\_text())

28w.writerow([blob1+&#39;. &#39;+blob,&#39;%s&#39; % (self.start)])

What this script does is take an object with a term, start date and end date and then inserting these keywords in the google search url where it will search for the top stories involving the term between the start date and end date. It then uses the requests library tool to go to the google url page and then uses beautifulSoup to save the html data of the request. When this is done each headline and summary on the google page are saved under the headline\_results and headline\_results1 as a list. A new csv file is then created and the data in the lists is then stored in the csv file by iterating through the list of news stories.

For each election, four different datasets were created with each focusing on a keyword in which top stories revolving around the keyword were scraped.

Below is a sample of one of the csv datasets which was scraped from google news data. The left column contains the news data while the right column contains the time period.

![](RackMultipart20210112-4-ahg56g_html_f966c4a6f218cffd.gif)

| **News Stories** | **Dates** |
| --- | --- |
| **Jeremy Corbyn increased Labour&#39;s vote share more than any of the party&#39;s**  **leaders since 1945. Jeremy Corbyn increased Labour&#39;s vote share more than any**  **of the party&#39;s leaders since 1945. Socialist achieves bigger swing than Tony Blair.**  **Harriet Agerholm ...** | **3A6%2F07%2F2017** |
| **Labour won social media election, digital strategists say. Party&#39;s use of Facebook**  **and Twitter to motivate voters, rather than attack Conservatives, regarded as key**  **to success. Robert Booth and Alex Hern. Fri 9 Jun 2017 ...** | **3A6%2F07%2F2017** |
| **Students inspired by Corbyn played big role in Labour surge. The surprise surge of**** Jeremy Corbyn&#39;s Labour party in the general election has been explained by a **** significant rise in youth turnout, with experts saying 18 to ... **|** 3A6%2F07%2F2017** |
| **Celebrate the transformation of the Labour Party, not the .... Lea Ypi argues that**  **the Labour Party is now a reinvigorated force. It is the largest social democratic**  **party in Europe; its members are now at the centre of its ...** | **3A6%2F07%2F2017** |
| **Gordon Brown tried to broker DUP deal after 2010 hung parliament. They&#39;ve** **effectively just replaced the Ulster Unionists (UUP) - but the two parties are**  **anything but the same. While the UUP is a relatively harmless centre-right party ...** | **3A6%2F07%2F2017** |
| **Thanks to Jeremy Corbyn, populism is no longer a dirty word. More than just**  **revive a moribund Labour Party and social democratic agenda – he is**  **fundamentally transforming who are the &quot;British people&quot;. Corbyn and**  **his ...** | **3A6%2F07%2F2017** |
| **Jeremy Corbyn, life and soul of the Labour Party. But although the personal**  **devotion that Mr Corbyn inspires among tented grime fans has given the party a**  **big boost, Labour remains divided on how it might ...** | **3A6%2F07%2F2017** |
| **Why the Rise of Corbyn&#39;s Labour Party Should Worry the West. In the days since**  **British Prime Minister Theresa May&#39;s disastrous snap election, the Labour Party**  **and its leader, Jeremy Corbyn, have been taking in the sheer ...** | **3A6%2F07%2F2017** |
| **Corbyn critics win election to Labour parliamentary committee. Ian Lavery, the**  **Labour party chairman, attempted to calm concerns among MPs on Friday, telling**** the Mirror: &quot;I don&#39;t see deselection as the way forward.&quot;. **|** 3A6%2F07%2F2017** |
| **The roar of the Glastonbury crowd is not enough – Labour .... If there were an**  **extended metaphor for the state of the Labour Party since the snap election, it**  **would be the Glastonbury Festival. I was there this year and heard ...** | **3A6%2F07%2F2017** |

Figure 4.2. Subset of Labour Party 2017 News Data

**4.1.1 Conservative Leader News**

This dataset consisted of top news stories of the conservative party candidate running for the position of Prime Minister of the UK. As the person running for this position may be different in each election, the candidates name is used as the keyword for retrieving the top news stories concerning them within the 6 month period pre elections. For each of the elections used in this projects these are the party leaders for the Conservative Party:

- 2005 - Michael Howard
- 2010 - David Cameron
- 2015 - David Cameron
- 2017 - Theresa May
- 2019 - Boris Johnson

**4.1.2 Conservative Party News**

This dataset consisted of top news stories of the conservative party as a whole. &#39;Conservative Party&#39;is used as the google search keyword for retrieving the top news stories concerning the party within the 6 month period pre elections.

**4.1.3 Labour Leader News**

This dataset consisted of top news stories of the labour party candidate running for the position of Prime Minister of the UK. As the person running for this position may be different in each election, the candidates name is used as the keyword for retrieving the top news stories concerning them within the 6 month period pre elections. For each of the elections used in this projects these are the party leaders for the Labour Party:

- 2005 - Tony Blair
- 2010 - Gordon Brown
- 2015 - Ed Miliband
- 2017 - Jeremy Corbyn
- 2019 - Jeremy Corbyn

**4.1.4 Labour Party News**

This dataset consisted of top news stories of the labour party as a whole. &#39;Labour Party&#39;is used as the google search keyword for retrieving the top news stories concerning the party within the 6 month period pre elections.

### **4.2 Sentiment Analysis**

Sentiment analysis measures the tone and intent of online comments. It involves natural language processing technologies to help comprehend entities and relationships to reveal positive, negative, neutral or uncertain attributes. An alternative approach to sentiment analysis includes more granular sentiment analysis which gives more precision in the level of polarity analysis which aims to identify emotions in expressions (e.g. happiness, sadness, frustration, surprise).

To perform sentiment analysis, the text has to be processed by performing the following[24]:

- Tokenization- Where text is split into sentences and individual words.
- Stemming- A method for classifying related words.
- Lemmatization- Words are reduced to their root form.
- Normalization- Transforming a list of words to a more uniform order.
- Vectorization- NLP methodology to convert words into numbers.

For my project I made use of the VADER( Valence Aware Dictionary for Sentiment Reasoning) which is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. It is available in the NLTK package and can be applied directly to unlabelled text data. VADER sentimental analysis relies on a dictionary that maps lexical features to emotion intensities known as sentiment scores. The sentiment score of a text can be obtained by summing up the intensity of each word in the text[25].

VADER was used to draw out the polarity in the political news datasets. The polarity lied between -1 to +1 with -1 being completely negative news and +1 for completely positive news. Each news story polarity was derived using the SentimentIntensityAnalyzer function in the nltk VADER library afterwards after deriving the polarity score of all the news stories and adding them to a list, the mean of all the polarity scores is calculated and this value is used to represent the total polarity score for the candidate or party for that particular election year.

1 sia = SentimentIntensityAnalyzer()

2 results = []

3

4 for text1,text in zip(headline\_results1,headline\_results):

5 blob1 = TextBlob(text1.get\_text())

6 blob = TextBlob(text.get\_text())

7 blob\_all = blob1+&#39;. &#39;+blob

8 pol\_score = sia.polarity\_scores(str(blob\_all))

9 pol\_score[&#39;headline&#39;] = str(blob\_all)

10 results.append(pol\_score)

11

12 dfr = pd.DataFrame.from\_records(results)

13 self.sentiment = dfr[&quot;compound&quot;].mean()

This script uses the SentimentIntensityAnalyzer function to calculate the polarity of the news data. It iterates through the list of news stories, calculates the polarity score of each one of them and then adds it to a new list. The mean of all the values are then calculated and saved as the sentiment polarity score of the object.

### **4.3 Constituency Socioeconomic Conditions Dataset**

The data gotten for this was gotten from the house of commons website which has made available variable collections of datasets for public usage. It uses 8 different socioeconomic categories to classify the 533 Westminster Parliamentary constituencies in England.

Below is a subset of all the data collated from the website

Figure 4.3 Subset of Constituency Socioeconomic Conditions Dataset part 1

Figure 4.4 Subset of Constituency Socioeconomic Conditions Dataset part 2

**4.3.1 Proportion Owning House**

![](RackMultipart20210112-4-ahg56g_html_33cdfb87e25a96bd.gif)
 ![](RackMultipart20210112-4-ahg56g_html_96ebc3cfee2e7fb.gif)

| **Constituency** | **House median**
 | **Wage Median**
 | **Proportion British**
 | **Prop owning**** own house **|** Cons. A\*-C % **|** 0-9 **|** 10-19** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Kensington** | 1450000 | 840 | 0.476823452 | 0.337770936 | 0.776 | 0.110977166 | 0.095173567 |
| **Cities of London and Westminster** | 1200000 | 780 | 0.480445455 | 0.315429136 | 0.741 | 0.091860223 | 0.087179448 |
| **Chelsea and Fulham** | 980000 | 830 | 0.566316925 | 0.403049714 | 0.764 | 0.119782414 | 0.089360611 |
| **Hampstead and Kilburn** | 755000 | 850 | 0.540614975 | 0.381704119 | 0.611 | 0.128909332 | 0.09497056 |
| **Holborn and St Pancras** | 751750 | 680 | 0.597108565 | 0.274977336 | 0.693 | 0.10442188 | 0.114619942 |
| **Westminster North** | 750000 | 830 | 0.464396126 | 0.304122651 | 0.735 | 0.129777765 | 0.104050337 |
| **Hammersmith** | 730000 | 800 | 0.553520078 | 0.307479118 | 0.641 | 0.118639502 | 0.093273278 |
| **Battersea** | 715000 | 830 | 0.654836987 | 0.419969195 | 0.801 | 0.117859055 | 0.071551819 |
| **Islington South and Finsbury** | 696250 | 870 | 0.637498052 | 0.265584498 | 0.717 | 0.097156457 | 0.089193218 |

| **20-29** | **30-39%** | **40-49%** | **50-59%** | **60-69** | **70-79%** | **80%** |
| --- | --- | --- | --- | --- | --- | --- |
| 0.140270139 | 0.175599454 | 0.151542083 | 0.129816534 | 0.091953012 | 0.068256413 | 0.036411633 |
| 0.183004473 | 0.196330826 | 0.14117004 | 0.120822523 | 0.087749147 | 0.057077749 | 0.034805571 |
| 0.157063316 | 0.179224894 | 0.149682229 | 0.12295097 | 0.083746955 | 0.062079892 | 0.03610872 |
| 0.156663317 | 0.20150809 | 0.147112979 | 0.106780335 | 0.074352351 | 0.054975146 | 0.03472789 |
| 0.224696255 | 0.175601215 | 0.121259665 | 0.106411286 | 0.074398785 | 0.04974144 | 0.028849533 |
| 0.146134176 | 0.205051552 | 0.14769974 | 0.110864266 | 0.073991516 | 0.049330162 | 0.033100487 |
| 0.185721502 | 0.197945782 | 0.145470618 | 0.114834147 | 0.071299882 | 0.046152551 | 0.026662738 |
| 0.210567026 | 0.254468801 | 0.134739777 | 0.094297444 | 0.05760355 | 0.037067888 | 0.02184464 |
| 0.273014749 | 0.209811172 | 0.115067568 | 0.096407072 | 0.058937851 | 0.037625687 | 0.022786228 |

This was a count of the percentage of the population who had ownership of their own residence. This has been included as a social and economic indicator. This data ranges from 97% in Sefton Central to 20% in Hackney South and Shoreditch and was assembled in the 2011 Census

**4.3.**** 2 **** Age **** D ****istribution**

This category splits the demographic of constituency residents into 9 age groups, (0-9, 10-19 and so on up until 80+) and has been included as a social indicator. This data was collected in the 2011 census.

**4.3.3 Proportion**** Born in the UK**

The percentage of residents in a constituency born in the UK has been included as a social indicator of a region. This data was collected in the 2011 Census and ranges from 41% in Brent North to 98% in Houghton and Sunderland South.

**4.3.4 Percentage of Students Achieving 5A\*-C Grades at GCSE**

The percentage of students achieving good school grades has been included as a social indicator. This data was collected in 2018 and ranges from 38% in Knowsley to 85% in Altrincham and Sale West.

**4.3.5 Median House Price**

Median house price is a measure of the cost of living in each constituency and has been included as an economic indicator. This data was recorded in March 2019 and ranges between £72,500 in Liverpool, Walton to £1,450,000 in Kensington.

**4.3.**** 6 Brexit Vote**

The Brexit category is an estimation of the percentage of residents in each constituency who voted to leave the European Union. This data is extrapolated from the 2016 Brexit referendum results and ranges from 20% in Hackney North and Stoke Newington to 76% in Boston and Skegness.

**4.3.7**  **Median Wage**

The weekly median salary of the population living in the constituency has been included as an economic indicator. This data was recorded in April 2019 and ranges from £890 in Wimbledon to £420 in Leicester East.

### **4.4 Election Result Dataset**

This dat was gotten from the house of commons website which has made available variable collections of datasets for public usage and wikipedia when it was unavailable on the house of commons website. Figure 4.5 below shows a subset of the election result data gotten from the website.

Figure 4.5 Subset of 2019 Election Results

### **4.5 Preprocessing**

![](RackMultipart20210112-4-ahg56g_html_94018f32351bc5e8.gif)

| **Constituency** | **Party** |
| --- | --- |
| **Aberavon** | Lab |
| **Aberconwy** | Con |
| **Aberdeen North** | SNP |
| **Aberdeen South** | SNP |
| **Airdrie and Shotts** | SNP |
| **Aldershot** | Con |
| **Aldridge-Brownhills** | Con |
| **Altrincham and Sale West** | Con |
| **Alyn and Deeside** | Lab |
| **Amber Valley** | Con |

Data preprocessing is an integral step in Machine Learning as the quality of data and the useful information that can be derived from it directly affects the ability of our model to learn; therefore, it is extremely important that we preprocess our data before feeding it into our model[26]. Data preprocessing in Machine Learning refers to the technique of preparing, cleaning and organising the raw data to make it suitable for a building and training Machine Learning models. In simple words, data preprocessing in Machine Learning is a [data mining technique](https://www.upgrad.com/blog/most-common-examples-of-data-mining/) that transforms raw data into an understandable and readable format.

**4.5.1 Data Quality Assessment**

Because my data was taken from multiple sources which sometimes were in different formats, a lot of time was consumed in dealing with data quality issues when working on this project. It was simply unfeasible to expect that the data will be perfect.

To fix the issue with missing values in the datasets, the affected rows in the tables are eliminated. This method works when there are not too many missing values in the data such that eliminating them will not lead to a huge change in results if the data was available. If a feature has mostly missing values, then that feature itself can also be eliminated. This issue arose with the 2005 election data because at that time there was a difference in the constituency structure compared to its present day arrangement. Because of that Constituency results that were expected in the dataset were missing as they were under different names or simply non-existing.

**4.5.2 Inconsistent values**

It is known that data can contain inconsistent values. This issue is likely to occur at some point. It may be due to human error or maybe the information was misread while being scanned from a handwritten form or just different datasets having different formats.

For my election result data the party in different elections were spelt differently. For example one dataset would refereed to the Liberal democrats party as &#39;Libs&#39; while another referred to them as &#39;LD&#39; while it was obvious for a human to know they referred to the same party, the computer would not be aware and see them as separate entities. Therefore all entries of the Liberal Democrats party was changed to &#39;LD&#39;.

**4.5.3 Feature Encoding**

Feature encoding is basically performing transformations on the data such that it can be easily accepted as input for machine learning algorithms while still retaining its original meaning. For nominal data a one to one mapping can be done. In the election result data, the winning party is nominal data and it has been preprocessed using the label encoder to become a numerical value. The political parties and their encoded values are:

- &#39;Con&#39; - 0
- &#39;Green&#39; - 1
- &#39;KHHC&#39; - 2
- &#39;LD&#39; - 3
- &#39;Lab&#39; - 4
- &#39;Resp&#39; - 5
- &#39;Spkr&#39; - 6
- &#39;UKIP&#39; - 7

### **4.6 Training**

### **4.7 Evaluation**

# **Chapter 5**

# **Implementation and Results**

# **Chapter 6**

# **Testing**

# **Chapter 7**

# **Conclusion**

# **Bibliography**

[1] BBC. (2019) General election 2019: What is an opinion poll? [online] Available at: www.bbc.co.uk/news/election-2019-50247874

[2] BBC. (2019) Exit polls: How accurate are they? [online] Available at: www.bbc.co.uk/news/election-2019-50716626

[3] Britannica. Parliamentary System.[online] Available at: www.britannica.com/topic/parliamentary-system

[4] Electoral Reform Society. (2017) First Past the Post. [online] Available at: www.electoral-reform.org.uk/voting-systems/types-of-voting-system/first-past-the-post

[5] Grofman, B., Trechsel, A. and Franklin, M., 2014. The internet and democracy in global perspective. _Voters, candidates, parties, and social movements introduction_.

[6] Dimova, G., 2019. Democracy Beyond Elections: Government Accountability in the Media Age. Springer Nature.

[7] Bimber, B. and Davis, R., 2003. Campaigning online: The Internet in US elections. Oxford University Press.

[8] MacLeod, A. ed., 2019. Propaganda in the information age: Still manufacturing consent. Routledge.

[9] Franch, F., 2013. (Wisdom of the Crowds) 2: 2010 UK election prediction with social media. _Journal of Information Technology &amp; Politics_, _10_(1), pp.57-71.

[10] Autoregressive integrated moving average.[online] Available at: [en.wikipedia.org/wiki/Autoregressive\_integrated\_moving\_average](http://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

[11] Sang, E.T.K. and Bos, J., 2012, April. Predicting the 2011 dutch senate election results with twitter. In _Proceedings of the workshop on semantic analysis in social media_ (pp. 53-60).

[12] What is Web Scraping ?[online] Available at: [www.webharvy.com/articles/what-is-web-scraping](http://www.webharvy.com/articles/what-is-web-scraping).

[13] Dogucu, M. and Çetinkaya-Rundel, M., 2020. Web Scraping in the Statistics and Data Science Curriculum: Challenges and Opportunities. _Journal of Statistics Education_, pp.1-11.

[14] Top 10 Best Scraping APIs.[online] Available at: [blog.api.rakuten.net/top-scraping-apis](http://blog.api.rakuten.net/top-scraping-apis)

[15] Dewi, L.C. and Chandra, A., 2019. Social Media Web Scraping using Social Media Developers API and Regex. _Procedia Computer Science_, _157_, pp.444-449.

[16] León-Borges, J.A., Noh-Balam, R.I., Gómez, L.R. and Strand, M.P., 2017. The machine learning in the prediction of elections-El aprendizaje automático en la predicción de las elecciones. _ReCIBE, Revista electrónica de Computació __n, Inform__ á __tica, Biom__ é __dica y Electró__ nica_, _4_(2).

[17] Tsakalidis, A., Papadopoulos, S., Cristea, A.I. and Kompatsiaris, Y., 2015. Predicting elections for multiple countries using Twitter and polls. _IEEE Intelligent Systems_, _30_(2), pp.10-17.

[18] Ramteke, J., Shah, S., Godhia, D. and Shaikh, A., 2016, August. Election result prediction using Twitter sentiment analysis. In _2016 international conference on inventive computation technologies (ICICT)_ (Vol. 1, pp. 1-5). IEEE.

[19] Sharma, P. and Moh, T.S., 2016, December. Prediction of Indian election using sentiment analysis on Hindi Twitter. In _2016 IEEE international conference on big data (big data)_ (pp. 1966-1971). IEEE.

[20] Garbade, M.J., 2018. A Simple Introduction to Natural Language Processing. Medium. Tillgängling online: https://becominghuman. ai/a-simple-introduction-to-naturallanguage-processing-ea66a1747b32 (2019-11-01).

[21] Oyebode, O. and Orji, R., 2019, October. Social media and sentiment analysis: The nigeria presidential election 2019. In _2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON)_ (pp. 0140-0146). IEEE.

[22] Digite. (2020). _What Is Kanban? An Overview Of The Kanban Method_. [online] Available at: https://www.digite.com/ kanban/what-is-kanban/

[23] Zenkit. (2020). _Agile Methodology: An Overview | Zenkit_. [online] Available at: https://zenkit.com/en/blog/agile- methodology-an-overview/

[24] Towardsdatascience. (2020). _&quot;Leveraging on NLP to gain insights in Social Media, News &amp; Broadcasting&quot;_ [online] Available at: [https://towardsdatascience.com/leveraging-on-nlp-to-gain-insights-in-social-media-news-broadcasting-ca89752ef638](https://towardsdatascience.com/leveraging-on-nlp-to-gain-insights-in-social-media-news-broadcasting-ca89752ef638)

[25] Towardsdatascience. (2020). _&quot;__SENTIMENTAL ANALYSIS USING VADER_ _interpretation and classification of emotions&quot;_ [online] Available at: [https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664](https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664)

[26] Towardsdatascience. (2018). &quot;Introduction to Data Preprocessing in Machine Learning&quot; [online] https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d
