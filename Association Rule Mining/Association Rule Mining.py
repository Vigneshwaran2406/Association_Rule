# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

# Import Dataset
book=pd.read_csv("D:\\book.csv")
print('First 5 Books Data')
print('-------------------')
print(book.head())

# With 10% Support
frequent_itemsets=apriori(book,min_support=0.1,use_colnames=True)
print('Association rules with 10% Support and 70% confidence')
print('-------------------------------------------------------')
print('With 10% Support')
print('=================')
print(frequent_itemsets)
# with 70% confidence
print('With 70% Confidence')
print('===================')
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
print(rules)
print(rules.sort_values('lift',ascending=False))
## A leverage value of 0 indicates independence. Range will be [-1 1]
## A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]
# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
print(rules[rules.lift>1])


# With 20% Support
print('Association rules with 20% Support and 60% confidence')
print('-------------------------------------------------------')
print('With 20% Support')
print('=================')
frequent_itemsets2=apriori(book,min_support=0.20,use_colnames=True)
print(frequent_itemsets2)
print('With 60% Confidence')
print('===================')
# With 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
print(rules2)


# With 5% Support
print('Association rules with 5% Support and 80% confidence')
print('-------------------------------------------------------')
print('With 20% Support')
print('=================')
frequent_itemsets3=apriori(book,min_support=0.05,use_colnames=True)
print(frequent_itemsets3)
# With 80% confidence
print('With 80% Confidence')
print('===================')
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
print(rules3)
print(rules3[rules3.lift>1])

# visualization of obtained rule
#Creating Subplots
figure,axis=plt.subplots(2,2)

#Plotting Association rules with 10% Support and 70% confidence
axis[0,0].scatter(rules['support'],rules['confidence'])
axis[0,0].set_title('Plotting with 10% Support & 70% confidence',fontsize=15)
axis[0,0].set_ylabel('confidence',fontsize=15)
axis[0,0].set_xlabel('support',fontsize=15)

#Plotting Association rules with 20% Support and 60% confidence
axis[0,1].scatter(rules2['support'],rules2['confidence'])
axis[0,1].set_title('Plotting with 20% Support & 60% confidence',fontsize=15)
axis[0,1].set_xlabel('support',fontsize=15)
axis[0,1].set_ylabel('confidence',fontsize=15)

#Plotting Association rules with 5% Support and 80% confidence
axis[1,0].scatter(rules3['support'],rules3['confidence'])
axis[1,0].set_title('Plotting with 5% Support & 80% confidence',fontsize=15)
axis[1,0].set_xlabel('support',fontsize=15)
axis[1,0].set_ylabel('confidence',fontsize=15)

#Adjusting Plots
plt.subplots_adjust(hspace=0.5, wspace=0.5, left=0.25, right=0.75)
figure.delaxes(axis[1,1])
figure.tight_layout()
plt.show()



