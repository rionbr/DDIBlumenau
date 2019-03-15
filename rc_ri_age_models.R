# Load lme3 Library for Random Effect
library(ggplot2)
library(stargazer)

# Set Working Directory
setwd('/Users/rionbr/Sites/DDIBlumenau/')
getwd()


# Load Data
data <- read.table('csv/age.csv', row.names=1, header=TRUE, sep=",", colClasses=c('character','integer','integer','integer','integer'))

# Summing 90+ group
new = data[0:18,]
new['90+',] = colSums(data[19:21,])
data = new

# Calculate RC & RI
data$RCy <- data$u..c. / data$u..n2.
data$RIy <- data$u..i. / data$u..c.
  
#
# RC
# Linear
RCy = data$RCy
y = c(0:(length(RCy)-1))
OLS.RCy.linear = lm(RCy ~ y, data=data.frame(RCy,y))
stargazer(OLS.RCy.linear, align=TRUE, single.row=TRUE, type='text')

# RC Quadratic
RCy = data$RCy
y = c(0:(length(RCy)-1))
OLS.RCy.quadratic = lm(RCy ~ I(y**2) + y, data=data.frame(RCy,y))
stargazer(OLS.RCy.quadratic, align=TRUE, single.row=TRUE, type='text')
anova(OLS.RCy.linear, OLS.RCy.quadratic)

# RC Cubic
RCy = data$RCy
y = c(0:(length(RCy)-1))
OLS.RCy.cubic = lm(RCy ~ I(y**3) + I(y**2) + y, data=data.frame(RCy,y))
stargazer(OLS.RCy.cubic, align=TRUE, single.row=TRUE, type='text')
anova(OLS.RCy.linear, OLS.RCy.quadratic, OLS.RCy.cubic)

#
# RI
# Linear
RIy = data$RIy
y = c(0:(length(RIy)-1))
OLS.RIy.linear = lm(RIy ~ y, data=data.frame(RIy,y))
stargazer(OLS.RIy.linear, align=TRUE, single.row=TRUE, type='text')

# RI Quadratic
RIy = data$RIy
y = c(0:(length(RIy)-1))
OLS.RIy.quadratic = lm(RIy ~ I(y**2) + y, data=data.frame(RIy,y))
stargazer(OLS.RIy.quadratic, align=TRUE, single.row=TRUE, type='text')
anova(OLS.RIy.linear, OLS.RIy.quadratic)

# RI Cubic
RIy = data$RIy
y = c(0:(length(RIy)-1))
OLS.RIy.cubic = lm(RIy ~ I(y**3) + I(y**2) + y, data=data.frame(RIy,y))
stargazer(OLS.RIy.cubic, align=TRUE, single.row=TRUE, type='text')
anova(OLS.RIy.linear, OLS.RIy.quadratic, OLS.RIy.cubic)

