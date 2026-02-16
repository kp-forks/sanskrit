library(data.table)
library(ggplot2)
library(MASS)
library(viridis)
library(PRROC)
library(stringr)
library(glmnet)
library(rstan)
library(ggrepel)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(shinystan)

# Standardization functions
z = function(x){
  (x-mean(x))/sd(x)
}

z.log = function(x){
  z(log(x))
}

z.mean = function(x){
  x-mean(x)
}

loglog = function(x,add.min = TRUE)
{
  if(min(x)==0){
    browser()
  }else if(min(x)==1){
    print("Case 1")
    y = log(log(1+x))
  }else{
    print("Case 2")
    y = log(log(x))
  }
  if(add.min){
    # "center" so that the largest class has value 0
    stopifnot(min(y)<0)
    y = y-min(y)
  }
  return(y)
}

#### 1. Prepare the data ####
data.path = "data-filtered-with-bert.csv"
data = fread(data.path,sep="\t",quote=FALSE,header=TRUE,data.table=FALSE,encoding='UTF-8')

#### Supplement this variable, it's central. ####
data$head.dir = ifelse(sign(data$distance)==1,"left","right")


####
max.sen.len = 50
data = data[data$sen.len<=max.sen.len,]


#### Head deprels ####
a = aggregate(data$head.deprel,list(Deprel=data$head.deprel),length)
rare = a$Deprel[which(a$x < 50)]
data$head.deprel[data$head.deprel %in% rare] = "other"

pos.lvls = c("NOUN",setdiff(unique(c(data$dep.upos,data$head.upos)),"NOUN"))
deprel.lvls = c("nmod","amod")


# Where is the dependent in its span?
drp = rep(0.5,nrow(data))
w = which(data$dep.start < data$dep.end)
drp[w] = (data$dep.position[w]-data$dep.start[w])/(data$dep.end[w]-data$dep.start[w])
dep.relpos = -1 + 2*drp
hrp = rep(0.5,nrow(data))
w = which(data$head.start < data$head.end)
hrp[w] = (data$head.position[w]-data$head.start[w])/(data$head.end[w]-data$head.start[w])
head.relpos = -1 + 2*hrp
txts = sprintf("%s %s",data$text,data$period)

#### Stan ####
# Submodel 1: arc lengths.
df.stan = data.frame(period=factor(as.integer(factor(data$period))),
                     poeticality=data$poeticality,
                     head.pos=factor(data$head.upos,levels=pos.lvls),
                     head.freq=z.log(1+data$head.frequency),
                     dep.pos=factor(data$dep.upos,levels=pos.lvls),
                     deprel=factor(data$deprel,levels=deprel.lvls),
                     dep.n.words = loglog(data$dep.n.words), 
                     dep.length.rank=log(1+data$dep.length.rank),
                     dep.freq=z.log(1+data$dep.frequency),
                     cossim=z(data$cossim), # !!
                     mi = z.mean(data$mi),
                     ### Attention: DO NOT standardize lpd.p, lpu.p, bert, at least
                     # not w/o thinking about it. Just using z.log etc. creates 
                     # super-divergent models.
                     lpd=data$lpd.p,
                     lpu=data$lpu.p
                     # End attention
)
df.stan$dep.length.rank = df.stan$dep.length.rank-min(df.stan$dep.length.rank)
stopifnot(all((df.stan$arclength+1) <= data$sen.len))
if("bert.prob" %in% colnames(data)){
  df.stan$bert = z.mean(data$bert.prob)
}


df.stan$arclength = log(abs(data$distance) - data$n.intruders)
df.stan$continuous = data$continuous

X = as.data.frame(model.matrix(continuous~., data=df.stan))

#### De-duplication ####
U = as.data.frame(apply(X,2,function(x)round(x,2)))
U$head.id = data$head.lemma.id
U$dep.id = data$dep.lemma.id
U$text = data$text
keys = apply(U,1,function(x)paste(x,collapse=" "))
dup = which(duplicated(keys))
print(sprintf("%d records duplicated",length(dup)),quote=FALSE)
sel = which(FALSE==duplicated(keys))
X = X[sel,]
data = data[sel,]
df.stan = df.stan[sel,]

#### Stratification factor ####
# Head dependency and directionality.
s.s = sprintf("%s %s", data$head.dir, data$head.deprel)
strata.lvls = sort(unique(s.s))
strata = as.integer(factor(s.s,levels=strata.lvls))
  


txt.fac = as.integer(as.factor(data$text))

stan.data.c = list(
  "N" = nrow(X),
  "O" = ncol(X),
  "continuous" = data$continuous,
  "X" = X,
  "strata" = strata,
  "nStrata" = max(strata),
  "texts" = txt.fac,
  "nTexts" = max(txt.fac)
)

model = "model"
init.r = 1.5 # Should be smaller than 2, else problems during initialization.
save.path = sprintf("%s.z",model)
mo = stan_model(file=sprintf('../stan/%s.stan',model))
z = stan( file=sprintf('../stan/%s.stan',model),
          data=stan.data.c,
          iter = 1500,
          chains= 3,
          refresh = 50, 
          init_r = init.r
)
saveRDS(list("z"=z,"stan.data"=stan.data.c,"df.stan"=df.stan,"data"=data),file=save.path)
