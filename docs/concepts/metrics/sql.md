# SQL 


## Execution based metrics
In these metrics the resulting SQL is compared after executing the SQL query on the database and then comparing the `response` with the expected results. 

### DataCompy Score

DataCompy is a python library that compares two pandas DataFrames. It provides a simple interface to compare two DataFrames and provides a detailed report of the differences. In this metric the `response` is executed on the database and the resulting data is compared with the expected data, ie `reference`. To enable comparison both `response` and `reference` should be in the form of a Comma-Separated Values as shown in the example.

Dataframes can be compared across rows or columns. This can be configured using `mode` parameter. 

If mode is `row` then the comparison is done row-wise. If mode is `column` then the comparison is done column-wise.

```{math}
:label: precision
\text{Precision } = {|\text{Number of matching rows in response and reference}| \over |\text{Total number of rows in response}|}
```

```{math}
:label: recall
\text{Precision } = {|\text{Number of matching rows in response and reference}| \over |\text{Total number of rows in reference}|}
```

By default, the mode is set to `row`, and metric is F1 score which is the harmonic mean of precision and recall.


```{code-block} python
from ragas.metrics._datacompy_score import DataCompyScore
from ragas.dataset_schema import SingleTurnSample

data1 = """acct_id,dollar_amt,name,float_fld,date_fld
10000001234,123.45,George Maharis,14530.1555,2017-01-01
10000001235,0.45,Michael Bluth,1,2017-01-01
10000001236,1345,George Bluth,,2017-01-01
10000001237,123456,Bob Loblaw,345.12,2017-01-01
10000001238,1.05,Lucille Bluth,,2017-01-01
10000001238,1.05,Loose Seal Bluth,,2017-01-01
"""

data2 = """acct_id,dollar_amt,name,float_fld
10000001234,123.4,George Michael Bluth,14530.155
10000001235,0.45,Michael Bluth,
10000001236,1345,George Bluth,1
10000001237,123456,Robert Loblaw,345.12
10000001238,1.05,Loose Seal Bluth,111
"""
sample = SingleTurnSample(response=data1, reference=data2)
scorer = DataCompyScore()
await scorer.single_turn_ascore(sample)
```
To change the mode to column-wise comparison, set the `mode` parameter to `column`.


```{code-block} python
scorer = DataCompyScore(mode="column", metric="recall")
```