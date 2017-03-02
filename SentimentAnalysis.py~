import nltk
from matplotlib import pyplot as plt
import numpy as np
from nltk import tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import csv
import numpy as np
import pandas as pd

# This is another test comment

names = ['data1',
         'data2',
         'data3']

speeches = ['/home/data1.txt',
             '/home/data2.txt',
             '/home/data3.txt']

# Initialise an empty dataframe to store the output
df = pd.DataFrame()
# Iterate over each sentiment text - calcualte Sentiments for each paragraph and store in online texts.
for name, speech in zip(names,speeches):
    rows_list=[]
    sid = SentimentIntensityAnalyzer()
    with open(speech) as csvfile:
        reader = csv.DictReader(csvfile,delimiter="\n",fieldnames = ['text'])
        for row in reader:
            #lines_list = tokenize.sent_tokenize(row['text'])
            # Line List Actually is ->  paragraph list
            lines_list = tokenize.sent_tokenize(row['text'].decode('unicode_escape').encode('ascii','ignore'))
            print lines_list
            score,pos,neg,neu = 0.0,0.0,0.0,0.0
            for sentence in lines_list:
                ss = sid.polarity_scores(sentence)
                print ss
                score = score + ss['compound']
                pos = pos + ss['pos']
                neg = neg + ss['neg']
                neu = neu + ss['neu']
                print ss, sentence, pos,neu,neg,score
            numofsents = len(lines_list)
            totalsentscompoundscore = score / numofsents
            totalsentspos = pos / numofsents
            totalsentsneu = neu / numofsents
            totalsentsneg = neg / numofsents
            print "total score: ", score, " numofsents:", numofsents, " sentcompound:",totalsentscompoundscore
            print "totalpos: ",totalsentspos," totalsentsneu: ",neu," totalsentsneg:",neg
            #agreggate everything into a dataframe row
            RecordtoAdd = {}  # initialise an empty dict
            RecordtoAdd.update({'{} numofsents'.format(name): numofsents})
            RecordtoAdd.update({'{} analyzedtext'.format(name): row['text']})
            RecordtoAdd.update({'{} sentcompound'.format(name): totalsentscompoundscore })
            rows_list.append(RecordtoAdd)

        analysed = pd.DataFrame(rows_list)
        pd.set_option('display.max_colwidth', -1)
        analysed = analysed.drop('{} analyzedtext'.format(name), axis=1)
        # df = df.join(analysed)
        df = pd.concat([df,analysed],axis = 1,)
        print df
    print analysed
    print 'here'
df.to_html('sentimentanalysisdf.html')


# 1 # no Scaling -> 3 subplots sharing the x axis
f,axs = plt.subplots(3,1,sharex = True)
print np.shape(axs)
# Iterate over all pairs of dataframe columns to plot
# There is probably a much nicer way to do this just via pandas plot
# Perhaps one to learn for the future
for i,name in enumerate(names):
    print name
    print i
    x = np.zeros(len(df.max(axis=1)))
    axs[i,].plot(x,color = 'gray', linestyle = '--',linewidth = 2)
    label = ['1','2','3']
    x = np.arange(0,len(df['{} sentcompound'.format(name)].dropna().values))
    axs[i,].plot(x,df['{} sentcompound'.format(name)].dropna().values,c = 'grey',  linewidth = 2, label = 't17',linestyle = ':')
    axs[i,].scatter(x,df['{} sentcompound'.format(name)].dropna().values,c = 'b', label = 't17')
    axs[i,].bar(x,df['{} sentcompound'.format(name)].dropna().values)
    axs[i,].set_title(name,fontsize = 8)
    axs[i,].set_ylabel('Emotional Valence',rotation = 90)
    axs[i,].set_xlabel('Narrative Time (Paragraph)')
    axs[i,].grid(b=True, which='major', color='black', alpha = 0.5, linestyle='-')
    axs[i,].set_ylim([-1,1])
    axs[i,].set_xlim([0,len(df['{} sentcompound'.format(name)])])

    #axs[i,].set_xlim([0, len(df['{} sentcompound'.format(name)].dropna().values)])

f.tight_layout()
# Turn on x labels for all
for i in axs:
    plt.setp(i.get_xticklabels(), visible=True)
f.suptitle('Semantic Analysis for Speeches',fontsize = 15,horizontalalignment = 'center')
plt.show()



# 2 # Scaled by Paragraph Number
f,axs = plt.subplots(3,1,sharex = True)
print np.shape(axs)
# Iterate over all pairs of dataframe columns to plot
# There is probably a much nicer way to do this just via pandas plot
# Perhaps one to learn for the future
for i,name in enumerate(names):
    print name
    print i
    x = np.zeros(len(df.max(axis=1)))
    axs[i,].plot(x,color = 'gray', linestyle = '--',linewidth = 2)
    label = ['1','2','3']

    # Carry out Scaling by Paragraph Number
    # Calculate the mean number of sentences (here it is 47) and scale the x values according to fit them into 47 paragraphs.
    constant = 47.
    # normalise = float(len(df.ix[:, 0].values)) / constant
    # Calculate the scaling factor for each list of values for each president.
    # Do not include NaN values (.dropna())
    normalise = float(len(df['{} sentcompound'.format(name)].dropna().values)) / constant
    print 'normalise:', normalise

    x = np.arange(0,len(df['{} sentcompound'.format(name)].dropna().values))
    x = x / normalise
    print 'x:',x

    axs[i,].plot(x,df['{} sentcompound'.format(name)].dropna().values,c = 'grey',  linewidth = 2, label = 't17',linestyle = ':')
    # axs[i,].scatter(range(0,len(df['{} sentcompound'.format(name)].dropna())),df['{} sentcompound'.format(name)].dropna().values,c = 'b', label = 't17')
    #axs[i,].scatter(range(0,len(df['{} sentcompound'.format(name)].dropna())),x,c = 'b', label = 't17')
    axs[i,].scatter(x,df['{} sentcompound'.format(name)].dropna().values,c = 'b', label = 't17')
    axs[i,].set_title(name,fontsize = 8)
    axs[i,].set_ylabel('Emotional Valence',rotation = 90)
    axs[i,].set_xlabel('Narrative Time (Scaled)')
    axs[i,].grid(b=True, which='major', color='black', alpha = 0.5, linestyle='-')
    axs[i,].set_ylim([-1,1])
    # axs[i,].set_xlim([0,len(df['{} sentcompound'.format(name)])])
    # When have scaling use:
    axs[i,].set_xlim([0, len(df['{} sentcompound'.format(name)].dropna().values)])

f.tight_layout()
# Turn on x labels for all
for i in axs:
    plt.setp(i.get_xticklabels(), visible=True)
f.suptitle('Semantic Analysis for Speeches -  Scaled by Paragraph No.',fontsize = 15,horizontalalignment = 'center')
plt.show()

# Scaled by NO. of Words per speech - this will change both overall scaling and scaling between paragraphs
# Each paragraph has a different number of words so the length will be different between each one
# Can use either a histogram or a bar chart to show this
# will be like an area spent talking with a particular level of sentiment.
