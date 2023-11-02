import sys, os, csv

os.chdir(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/data/')

with open('ClaimReview-COVID_output.csv', 'r') as fout:
    claimreview=fout.read()
    #for line in fout:
newlines=claimreview.split("http://data.cimple.eu/")
data=dict()
for record in newlines[1:]:
    data['"'+record.split(',')[0]+'"']=','.join(record.split(',')[2:]).replace('\n','')


with open('claimreviews.csv', 'w') as fout:
    for i in data.keys():
        fout.write(i+','+data[i]+'\n')
    #writer = csv.writer(fout, quoting=csv.QUOTE_ALL, quotechar='"')
    #writer.writerow(data.keys())
    #writer.writerows(zip(*data.values()))

