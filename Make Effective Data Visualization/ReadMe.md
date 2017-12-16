# Make Effective Data Visualization

### Summary

This project visualized baseball dataset and explored what factor will impact a player's performance.
The performace can be valued by `avg (batting average)`. Factors like `handedness`
and `height` may have an impact on the performance. 
The visualized data shows that player with low height value tends to have high average batting value, that is to say, a better performance.

### Design

Since the dataset is small, all data can be included in the plot.
I choose bubble plot to visualize the dataset.
X-axis is `height` and Y-axis is `avg`.
The category `handedness` can be shown by different color.
The value of `HR` is displayed through the size of the bubbules.

### Feedback


__feedback 1__

I think the plot is not clear engouth. It's a little bit overplot.
Althogh I can get the information of a particular data point, it's hard for me to find a general trend from this plot.
Maybe you can first aggregate the data by group like "height" or "handedness" and then plot these data points.


__feedback 2__

The visualization would be better if you implemented animation.
I also recommand to add a line on the bubble. The trend of different handedness will be more clear.

__feedback 3__

I like your visualization. I think the narrative is supported by the data.
I might consider to give the users some choices about what data they want to see.
Perhaps it would be help to add a filter which enables users to choose whether they want to view right-handedness player or left-handedness player.

__feedback 4__

The HR value has the same importance with avg. The HR value shown by bubble size is less perceptual than avg value shown by y-axis. The count of each category is more suitble to be displayed by bubble size. 
It would be better to show the square root of the count to display the plot properly.

__my change__

I accepted all advice above and changed my plot. The total number of the dataset is 1156. It would cause overplot if I plot all data points. So I use the average value to represent a group of data. Batting average have the same importance with home run, so I only choose batting average to show the players' performance, which will make the plot more clear. The bubble size shows the count of each group of data.

### Resource

[Time Bubble Lines](http://dimplejs.org/advanced_examples_viewer.html?id=advanced_time_axis)

[Tooltip Charts](http://dimplejs.org/advanced_examples_viewer.html?id=advanced_lollipop_with_hover)

[d3-3.x-api-reference](https://github.com/d3/d3-3.x-api-reference/blob/master/API-Reference.md)

[dimple wiki](https://github.com/PMSI-AlignAlytics/dimple/wiki)