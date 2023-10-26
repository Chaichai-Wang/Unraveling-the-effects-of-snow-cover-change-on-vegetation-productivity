library("plspm")
litter <- read.csv(file.choose(),header = T)

#Phen作为潜变量
SWE<-c(0,0,0,0,0,0,0,0,0,0)
SCD<-c(0,0,0,0,0,0,0,0,0,0)
SCED<-c(0,0,0,0,0,0,0,0,0,0)
Prec<-c(0,0,0,0,0,0,0,0,0,0)
Temp<-c(0,0,0,0,0,0,0,0,0,0)
Srad<-c(0,0,0,0,0,0,0,0,0,0)
SoilMos<-c(1,1,1,1,0,0,0,0,0,0)
SoilTem<-c(1,1,1,0,1,0,0,0,0,0)
Phen<-c(1,1,1,0,0,0,0,0,0,0)
GPP<-c(0,0,0,1,1,1,1,1,1,0)

#Phen作为潜变量
dat_path <- rbind(SWE,SCD,SCED,Prec,Temp,Srad,SoilMos,SoilTem,Phen,GPP)

colnames(dat_path) <- rownames(dat_path)
#dat_path
innerplot(dat_path,arr.pos =0.25)

dat_blocks <- list(
  SWE='SWE',
  SCD='SCD',
  SCED='SCED',
  Prec='Prec',
  Temp='Temp',
  Srad='Srad',
  SoilMos='SMois',
  SoilTem='STemp',
  #生长季早期
  Phen=c('UD','SD'),
  #生长季峰值期
  #Phen=c('SD','DD'),
  GPP='GPP'
)

dat_modes <- rep('A',10)
dat_pls <- plspm(litter, dat_path,dat_blocks,modes = dat_modes)
#summary(dat_pls)
innerplot(dat_pls, colpos = 'red', colneg = 'blue', show.values = TRUE, lcol = 'BLACK', box.lwd = 0,arr.pos =0.25,dtext=0.1)
dat_pls$gof
dat_pls$path_coefs
dat_pls$inner_model
dat_pls$outer_model
