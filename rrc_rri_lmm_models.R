ggCaterpillar <- function(re, QQ=TRUE, likeDotplot=TRUE, reorder=TRUE) {
  require(ggplot2)
  f <- function(x) {
    pv   <- attr(x, "postVar")
    cols <- 1:(dim(pv)[1])
    se   <- unlist(lapply(cols, function(i) sqrt(pv[i, i, ])))
    if (reorder) {
      ord  <- unlist(lapply(x, order)) + rep((0:(ncol(x) - 1)) * nrow(x), each=nrow(x))
      pDf  <- data.frame(y=unlist(x)[ord],
                         ci=1.96*se[ord],
                         nQQ=rep(qnorm(ppoints(nrow(x))), ncol(x)),
                         ID=factor(rep(rownames(x), ncol(x))[ord], levels=rownames(x)[ord]),
                         ind=gl(ncol(x), nrow(x), labels=names(x)))
    } else {
      pDf  <- data.frame(y=unlist(x),
                         ci=1.96*se,
                         nQQ=rep(qnorm(ppoints(nrow(x))), ncol(x)),
                         ID=factor(rep(rownames(x), ncol(x)), levels=rownames(x)),
                         ind=gl(ncol(x), nrow(x), labels=names(x)))
    }
    
    if(QQ) {  ## normal QQ-plot
      p <- ggplot(pDf, aes(nQQ, y))
      p <- p + facet_wrap(~ ind, scales="free")
      p <- p + xlab("Standard normal quantiles") + ylab("Random effect quantiles")
    } else {  ## caterpillar dotplot
      p <- ggplot(pDf, aes(ID, y)) + coord_flip()
      if(likeDotplot) {  ## imitate dotplot() -> same scales for random effects
        p <- p + facet_wrap(~ ind)
      } else {           ## different scales for random effects
        p <- p + facet_grid(ind ~ ., scales="free_y")
      }
      p <- p + xlab("Levels of the Random Effect") + ylab("Random Effect")
    }
    
    p <- p + theme(legend.position="none")
    p <- p + geom_hline(yintercept=0)
    p <- p + geom_errorbar(aes(ymin=y-ci, ymax=y+ci), width=0, colour="black")
    p <- p + geom_point(aes(size=1.2), colour="blue") 
    return(p)
  }
  
  lapply(re, f)
}

#
# Date: Nov, 14, 2016
# Author: Rion Brattig Correia
#
# Description: Performs Random Effect Models (Mixed Models) on the DDI_BRAZIL dataset
#

# Install Packages if you don't have them
install.packages("lme4")
install.packages("nlme")
install.packages("ggplot2")
install.packages("car")
install.packages("stargazer")

# Load lme3 Library for Random Effect
library(ggplot2)
library(lme4)
library(car)
library(stargazer)

# Set Working Directory
setwd('/Users/rionbr/Sites/DDIBlumenau/')
getwd()

# Load Data
data <- read.table('csv/df_statistical_modeling.csv', header=TRUE, sep=",")

data <- within(data, escolaridade <- relevel(escolaridade, ref='Not reported'))

# Remove Bairro Other
dataBairros = data[ which(data$bairro!="OTHER") , ]
# Only data with >1 interaction
dataInter = data[ which(data$n_ij_ddi>0) , ]
dataCoAdmin = data[ which(data$n_ij>0) , ]
# Remove NotReported Education
dataEdu = data[ which(data$education!="Not reported") , ]
# Remove NotInformed EstadoCivil
dataEst = data[ which(data$estado_civil!="Not informed" & data$estado_civil!="Ignored") , ]
# Age Brackets
dataAgeM39 = data[ which(data$age<=39) , ]
dataAgeP40 = data[ which(data$age>=40) , ]

#SR Formulas
f1l  = n_ij ~ n_i
f1c  = n_ij ~ n_i + I(n_i^2)
f2l  = n_ij_ddi ~ n_i
f2c  = n_ij_ddi ~ n_i + I(n_i^2)
f3l  = n_ij_ddi ~ n_ij
f3c  = n_ij_ddi ~ n_ij + I(n_ij^2)

# Simple Regression
OLS.f1l   <- lm(f1l, data=data); stargazer(OLS.f1l, align=TRUE, single.row=TRUE, type='text');
OLS.f1c   <- lm(f1c, data=data); stargazer(OLS.f1c, align=TRUE, single.row=TRUE, type='text');
OLS.f2l   <- lm(f2l, data=data); stargazer(OLS.f2l, align=TRUE, single.row=TRUE, type='text');
OLS.f2c   <- lm(f2c, data=data); stargazer(OLS.f2c, align=TRUE, single.row=TRUE, type='text');
OLS.f3l   <- lm(f3l, data=data); stargazer(OLS.f3l, align=TRUE, single.row=TRUE, type='text');
OLS.f3c   <- lm(f3c, data=data); stargazer(OLS.f3c, align=TRUE, single.row=TRUE, type='text');

#MR Formulas
fb   = n_ij_ddi ~ n_i + n_ij
fbt1 = n_ij_ddi ~ n_i + n_ij + I(n_i^2)
fbt2 = n_ij_ddi ~ n_i + n_ij + I(n_ij^2)
fbt3 = n_ij_ddi ~ n_i + n_ij + I(n_i^2) + I(n_ij^2)
#
fag  = n_ij_ddi ~ n_i + n_ij + age + C(gender)
fag2 = n_ij_ddi ~       n_ij + age
fel  = n_ij_ddi ~ n_i + n_ij + C(education)
felt = n_ij_ddi ~ n_i + n_ij + C(education_tnc)
fm   = n_ij_ddi ~ n_i + n_ij + C(marital)
fi   = n_ij_ddi ~ n_i + n_ij + avg_income
fnei = n_ij_ddi ~ n_i + n_ij + C(hood)
fother = n_ij_ddi ~ n_i + n_ij + theft_pc + robbery_p1000 + suicide_p1000 + transitcrime_p1000 + traffic_p1000 + rape_p1000
#

# Multiple Regression 
# Baseline
OLS.b   <- lm(fb, data=data); stargazer(OLS.b, align=TRUE, single.row=TRUE, type='text');
OLS.bt1 <- lm(fbt1, data=data); stargazer(OLS.bt1, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.bt1)
OLS.bt2 <- lm(fbt2, data=data); stargazer(OLS.bt2, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.bt2)
OLS.bt3 <- lm(fbt3, data=data); stargazer(OLS.bt3, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.bt3)
#
OLS.ag  <- lm(fag, data=data); stargazer(OLS.ag, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.ag)
OLS.ag2  <- lm(fag2, data=data); stargazer(OLS.ag2, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.ag2)
#
OLS.e   <- lm(fb, data=dataEdu); stargazer(OLS.e, align=TRUE, single.row=TRUE, type='text');
OLS.el  <- lm(fel, data=dataEdu); stargazer(OLS.el, align=TRUE, single.row=TRUE, type='text'); anova(OLS.e,OLS.el)
OLS.elt <- lm(felt, data=dataEdu); stargazer(OLS.elt, align=TRUE, single.row=TRUE, type='text'); anova(OLS.e,OLS.elt)
#
OLS.m   <- lm(fm, data=data); stargazer(OLS.m, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.m)
OLS.i   <- lm(fi, data=data); stargazer(OLS.i, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.i)
OLS.nei <- lm(fnei, data=data); stargazer(OLS.nei, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.nei)
OLS.other <- lm(fother, data=data); stargazer(OLS.other, align=TRUE, single.row=TRUE, type='text'); anova(OLS.b,OLS.other)
#
OLS.M39.b <- lm(fb, data=dataAgeM39); stargazer(OLS.M39.b, align=TRUE, single.row=TRUE, type='text');
OLS.M39.ag <- lm(fag, data=dataAgeM39); stargazer(OLS.M39.ag, align=TRUE, single.row=TRUE, type='text'); anova(OLS.M39.b,OLS.M39.ag)
#
OLS.P40.b <- lm(fb, data=dataAgeP40); stargazer(OLS.P40.b, align=TRUE, single.row=TRUE, type='text');
OLS.P40.ag <- lm(fag, data=dataAgeP40); stargazer(OLS.P40.ag, align=TRUE, single.row=TRUE, type='text'); anova(OLS.P40.b,OLS.P40.ag)

#
# Random Mixing Effect (LMM)
#
LMM.hood <- lmer(n_ij_ddi ~ n_i + n_ij + (1|hood), data=data, REML=FALSE); summary(LMM.h)
LMM.nested <- lmer(n_ij_ddi ~ n_i + n_ij + (1|age/gender), data=data, REML=FALSE)

