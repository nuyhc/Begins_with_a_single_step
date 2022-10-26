# Plotly
[Plotly](https://plotly.com/python/)

```python
import plotly.express as px
```

## Plotly 예제 - 금융 데이터 시각화


```python
df = px.data.stocks()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>GOOG</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>NFLX</th>
      <th>MSFT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-08</td>
      <td>1.018172</td>
      <td>1.011943</td>
      <td>1.061881</td>
      <td>0.959968</td>
      <td>1.053526</td>
      <td>1.015988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-15</td>
      <td>1.032008</td>
      <td>1.019771</td>
      <td>1.053240</td>
      <td>0.970243</td>
      <td>1.049860</td>
      <td>1.020524</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-22</td>
      <td>1.066783</td>
      <td>0.980057</td>
      <td>1.140676</td>
      <td>1.016858</td>
      <td>1.307681</td>
      <td>1.066561</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-29</td>
      <td>1.008773</td>
      <td>0.917143</td>
      <td>1.163374</td>
      <td>1.018357</td>
      <td>1.273537</td>
      <td>1.040708</td>
    </tr>
  </tbody>
</table>
</div>



### 일별 수익률 선그래프 그리기
선 그래프는 `px.line`을 사용한다  
`px.line(data, x=, y=, title=)`이 가장 기본적인 형태인거 같다


```python
px.line(df, x="date", y="GOOG", title="구글 주가")
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/7.embed"></iframe>

### 일별 수익률 막대그래프 그리기
막대 그래프는 `px.bar`을 사용한다.  
df의 `date` 컬럼을 인덱스로 변경하고, 첫 데이터를 0으로 만들어 기준점을 만들어 그래프를 그렸다


```python
df_bar = df.set_index("date") - 1
df_bar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOOG</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>NFLX</th>
      <th>MSFT</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>0.018172</td>
      <td>0.011943</td>
      <td>0.061881</td>
      <td>-0.040032</td>
      <td>0.053526</td>
      <td>0.015988</td>
    </tr>
    <tr>
      <th>2018-01-15</th>
      <td>0.032008</td>
      <td>0.019771</td>
      <td>0.053240</td>
      <td>-0.029757</td>
      <td>0.049860</td>
      <td>0.020524</td>
    </tr>
    <tr>
      <th>2018-01-22</th>
      <td>0.066783</td>
      <td>-0.019943</td>
      <td>0.140676</td>
      <td>0.016858</td>
      <td>0.307681</td>
      <td>0.066561</td>
    </tr>
    <tr>
      <th>2018-01-29</th>
      <td>0.008773</td>
      <td>-0.082857</td>
      <td>0.163374</td>
      <td>0.018357</td>
      <td>0.273537</td>
      <td>0.040708</td>
    </tr>
  </tbody>
</table>
</div>




```python
px.bar(df_bar)
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/9.embed"></iframe>


### 서브플롯 그리기
`facet_col` 옵션을 이용해 서브플롯을 그릴수 있다.  
`facet_col` 옵션에 구분할 컬럼의 이름을 넣고, `facet_col_wrap`과 `facet_row_wrap` 옵션을 이용해 크기도 조절 가능하다.


```python
df_bar.columns.name = "company"
df_bar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>company</th>
      <th>GOOG</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>NFLX</th>
      <th>MSFT</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>0.018172</td>
      <td>0.011943</td>
      <td>0.061881</td>
      <td>-0.040032</td>
      <td>0.053526</td>
      <td>0.015988</td>
    </tr>
    <tr>
      <th>2018-01-15</th>
      <td>0.032008</td>
      <td>0.019771</td>
      <td>0.053240</td>
      <td>-0.029757</td>
      <td>0.049860</td>
      <td>0.020524</td>
    </tr>
    <tr>
      <th>2018-01-22</th>
      <td>0.066783</td>
      <td>-0.019943</td>
      <td>0.140676</td>
      <td>0.016858</td>
      <td>0.307681</td>
      <td>0.066561</td>
    </tr>
    <tr>
      <th>2018-01-29</th>
      <td>0.008773</td>
      <td>-0.082857</td>
      <td>0.163374</td>
      <td>0.018357</td>
      <td>0.273537</td>
      <td>0.040708</td>
    </tr>
  </tbody>
</table>
</div>




```python
px.bar(df_bar, facet_col="company", facet_col_wrap= 2)
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/9.embed"></iframe>


`px.area`를 이용하면 분포를 그릴 수 있다.


```python
px.area(df_bar, facet_col="company", facet_col_wrap=2)
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/12.embed"></iframe>


### Range Slider
그래프에 `.update_xaxes(rangeslider_visible=True)`을 붙여주면 서브플롯 느낌으로 하단에 슬라이더가 생성된다.


```python
px.line(df_bar, x=df_bar.index, y="GOOG", title="GOOG 주가").update_xaxes(rangeslider_visible=True)
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/14.embed"></iframe>


### Scatterplot
`px.scatter_matrix`로 그릴 수 있다.


```python
px.scatter_matrix(df_bar)
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/15.embed"></iframe>


### 분포 그리기
#### box
`px.box`로 그릴 수 있다.  
`points` 옵션을 이용하면 모든 분포가 나오고, `notched` 옵션을 이용하면 박스의 형태가 분포 형태에 따라 변경된다.  
`color` 옵션의 경우 범주형 데이터를 이용해 색을 구분할 수 있게 해준다.  

하나의 데이터에 대해서만 그릴 경우, `x`에 값을 지정해줘서 사용 가능하다.


```python
px.box(df_bar, color="company")
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/17.embed"></iframe>



```python
px.box(df_bar, points="all", notched=True, color="company")
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/17.embed"></iframe>


#### violin
`px.violin`을 이용해 그릴 수 있고, 박스와 유사한 옵션들을 사용할 수 있다.


```python
px.violin(df_bar, points="all", box=True, color="company")
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/20.embed"></iframe>


#### strip
`px.strip`을 이용해 그릴 수 있다.


```python
px.strip(df_bar, color="company")
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/22.embed"></iframe>


#### histogram
`px.histogram`으로 그릴 수 있다.

```python
px.histogram(df_bar, marginal="box", facet_col="company")
```
<iframe width="100%" height="525" frameborder="0" scrolling="no" src="//plotly.com/~nuyhc/24.embed"></iframe>

