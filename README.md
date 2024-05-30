# Forudsigelse af kraniofacial vækst og tandstilling forlæbe-ganespaltepatienter


## Indholdsfortegnelse
* [Introduktion](#introduktion)
* [Model](#model)
* [SHAP værdier og hvad bruger vi dem til.](#shap-værdier-og-hvad-bruger-vi-dem-til)
* [Udfordringer](#udfordringer)
* [Andre tilgange](#andre-tilgange)
* [Konklusion](#konklusion)
* [Referencer](#referencer)

# Introduktion
Læbe-ganespalte er en medfødt tilstand, som rammer omkring 1 / 500 af børn. Børn med læbe-ganespalte gennemgår tre operationer (se figur 1). En primær operation, som er en kirurgisk lukning af læbespalten og den bløde gane. Dette sker, når patienterne er spædbørn. I en alder af enten et eller tre år lukkes den hårde gane. Når patienterne er 8 år, lukkes alveolær spalte (spalte i gummen), og bøjlebehandlingen påbegyndes. I 12-års alderen vil den endelige bøjlebehandling blive påbegyndt, og der vil eventuelt blive foretaget mindre justeringer. Til sidst vil patientens bøjle blive taget af, typisk i 16-års alderen.

<img width="894" alt="Skærmbillede 2024-04-03 kl  22 26 42" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/ebdc42f4-f83a-4bb9-ada1-b10800b02e95">\
(figur.1: forløb for ganespaltepatienter)\
I et internationalt studie, har man undersøgt forskellige primære operationer til sammenligning. I forbindelse med dette har man i Danmark indsamlet data på børnenes udvikling ved henholdsvis 8, 12 og 16 år.

 Datasættet har 124 patienter med 36 kolonner, og består af tre målings-tidspunkter i hhv. aldrene 8, 12 og 16. For hver måling har vi 10 værdier for patientens tilstand, f.eks. 'Spacing, 'Transverse' og 'Crowding'. Summen af alle værdierne udgør patientens Pinheiro score. Scoren ligger mellem 0 og 52, og man ønsker en lav score.

Vores model, som tager værdier fra målinger af tændernes tilstand efter henholdsvis 8 og 12 år, og vil prædiktere den endelige tilstand når patienten er 16 år, baseret på tidligere datapunkter
(Denne model bliver lavet til en app som tandlægerne kan bruge som et værktøj til at bekræfte deres faglige intuition.)

For at tandlæger kan tilgå og bruge modellen i praksis, er der udviklet en app igennem Streamlit. Gennem appen kan tandlægerene  indtaste værdier for de første to målinger, hvortil vores model vil give en prædiktion for den endelige udvikling, samt hvor sikker modellen er på sin forudsigelse. 


# Model 

## Hvilken model bruger vi, og hvorfor.
Vi har valgt at bruge en logistisk regression, da vi ønskede at lave en binær klassificering, og med den mængde data vi har, gav det mening at bruge en supervised model, hvor labels er god eller dårlig. Den logistiske regressions output er en sandsynlighed for at ende i en god eller dårlig.
For bedre at kunne demonstrere, hvordan vi bruger dette i praksis, gennemgår vi kort teorien bag den multiple logistisk regression, samt hvordan vi implementerer dette i Python.

## Hvad indebærer modellen, og hvordan den virker i praksis.

Givet vi har n datapunkter, som er I.I.D og er angivet på formen $X = [x_1,x_2,...,x_n]$. Da er logit af vores multipel regression givet som: 
$$g(x) =\ln\left(\frac{\pi(X)}{1-\pi(X)}\right) = \beta_0+\beta_1x_1+...+b_n x_n$$
hvor $n=16$.

Vi kan nu opskrive vores multipel logisitiske regression på formen:

$$\pi(x)=\frac{e^{g(x)}}{1+e^{g(x)}}$$

Vi vil nu opskrive vores likelihood funktion. Vi har et $Y$, vores "dummy variabel", som er 0 eller 1. $\pi(x)$ er en betinget sandsynlighed, som er $P(Y=1|x) = \pi(x)$, hvis $Y = 1$ og $1-\pi(x)$, hvis $Y = 0$ . Vi kan nu benytte Bernoulli fordelingen til at opstille vores likelihood funktion.

$$\pi(x_i)^{y_i}(1-\pi(x_i))^{1-y}$$
Vi ved at alle observationer er uafhængige, da vores likelihood er et samlet produkt af udtrykket oven over.

$$l(\beta')=\prod^n_{i=1} \pi(x_i)^{y_i} (1-\pi(x_1))^{1-y}$$
hvor $\beta' = [\beta_0,...,\beta_m]$
Princippet bag maksimum likelihood funktionen er at estimere værdien for hver $\beta_m$, som maksimerer udtrykket. For at gøre det nemmere at estimere $\beta'$, benytter vi Log-likelihood methoden og omskriver  $l(\beta')$ til 

$$L(\beta')=\ln(l(\beta')) = \sum^n_{i=1} y_i \ln(\pi(x_i))+(1-y_i)\ln(1-\pi(x_i))$$

Nu differencierer vi $L(\beta')$ med respekt til $\beta'$ for at finde de værdier, som maksimerer vores udtryk
$$\hat{\beta'}= \frac{\partial L(\beta')}{\partial \beta_m} = \sum^n_{i=1} y_i x_{im} - x_{im} \pi(x_i) = 0$$
Nu har vi fundet vores maksimum likelihood estimater, som vi beskriver ved $\hat{\beta}$

Nu hvis vi gerne vil lave en forudsigelse med vores model, benytter vi vores $\hat{\beta'}$ og indsætter dem i 
$$\pi(x) = \frac{e^{\hat{\beta_0}+\hat{\beta_1}x_1+...+\hat{\beta_m}x_m}}{1+e^{\hat{\beta_0}+\hat{\beta_1}x_1+...+\hat{\textbf{\beta'}}x_m}}$$


Alt teorien er fundet i (1) s.6-9 og s.31-34


I Jupyter notebook filen 'Logistisk-regression.ipynb' bruger vi Sklearn til lave og træne vores model. Helt generelt så importerer vi det behandlet data, da standardiserer vi data med en MinMax, hvilket betyder at datapunkter bliver skaleret til at være mellem 0 og 1. Da kan vi oprette vores binær kolonne. Dernæst splitter vi data op i 75 procent trænings data og 25 procent test data. Nu kan vi træne vores model og bagefter bruge test data til analysere hvor præcis vores model er, vi laver også en confusion matrix for at se, hvordan vores model gætter. Da vi havde store svingninger i vores models præcision, valgte vi at bruge en Bootstrap tilgang, hvor vi kører modellen 25 gange og tager middelværdien af forudsigelserne, præcision og SHAP-værdierne. Det har gjort at vores model er mere stabil og giver en mere ensartet forudsigelse.

## Streamlit app 
En del af projekt var at finde en måde hvormed tandlægerne kunne benytte vores model, uden at have kendskab til Python eller den bagved liggende matematik, det endte ud i en app, hvor tandlægerne kan bruge den model vi har lavet som et værktøj. Det er gjort nemt for dem at indsætte tal og for en forudsigelse tilbage. For at lave appen har vi brugt python pakken ‘Streamlit’ som producerer en cloud-based hjemmeside.

Appen er bruger vores bootstrap logistisk regression som i ‘logistisk regression.ipynb’ filen, og producerer præcision, spredning og SHAP værdier for modellen udefra det.

Det betyder at vi laver et slags dashboard, det gør det nemt for tandlægerne at bruge vores model uden at skulle installere python eller overhovedet forstå det tekniske bag vores model og forudsigelse. 
Hjemmesiden er blevet modtaget med stor glæde og interesse fra tandlægerne. 
 
Link til hjemmeside:  https://cleft-lip-app-r4y7280urvh.streamlit.app

 
## SHAP værdier og hvad bruger vi dem til.
SHAP (Shapley Additive exPlanations) er en metode, der kan bruges på Machine Learning modeller, til at se hver parameters effekt på en forudsigelse.\
Når man arbejder med SHAP-værdier, er det vigtigt at notere sig, at de ikke kan bruges til at forklare kausalitet. Siger udelukkende noget om, hvordan modellen er kommet frem til en forudsigelse.\
\
Vi bruger to plots fra pakken SHAP, til forklare modellens forudsigelser. Det er muligt at se hvordan SHAP værdier bliver lavet i 'logistisk regression.ipynb'. Her kan man se, at de fleste gange modellen bliver kørt, vil Antereoposterior 2.1 være den parameter med størst effekt.

![image](https://github.com/Christofferfuglkjaer/Dataproject/assets/118052934/026fdeb7-9239-4f76-a84a-c99aae7f6f91)\
(figur 2. SHAP plot)

## Udfordringer 

På grund af vores data har vi haft nogle problemer med at lave vores model. Vi har været begrænset af størrelsen af vores data, der indeholder 123 patienter, som har gjort at vi har skulle være opmærksomme på overfitting. Yderlige har der været datapunkter med manglende værdier, så vi ender med 116 datapunkter. Datoerne for målingerne er tilgængelige for patienter fra hovedstaden, men manglende for patienter fra Aarhus, hvilket tvinger os til ikke at bruge dem. De kunne have haft en effekt, da patientens alder ved sidste varierer en del. Dette skyldes, at patienten skal være helt færdig med behandlingen før de får taget de sidste mål.

Begrænsningen af et lille datasæt har vi forsøgt at undgå i vores logistiske regressionsmodel ved at bruge Bootstrapping. Her vil vi lave modellen på forskellige dele af datasættet, hvor den gennemsnitlige model da gerne skulle have mindre overfitting.


Derudover har datasættet ikke et stærkt sammenhæng mellem målingerne fra 8 og 12 år og målingerne fra 16 år. Særligt har vi set, at patienter, som har en høj score i de to første målinger, kan ende med både gode og dårlige resultater i sidste måling, som kan ses i figuren under.

<img width="730" alt="image" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/ac63ddf7-863b-44bc-9182-4893c11ec845">\
(figur 3. )

For patienter med end Pinheiro score over 20 i målingen ved 12 år ender omkring 37\% af dem med et godt resultat. Det giver en del usikkerhed i vores model, som derfor har en tendens til at gætte imellem. I denne figur har vi lavet modellem 10.000 gange med et udsnit af datasættet som træningsdata, og observeret hvor sikker den er på, at hver testpatient ender med et godt resultat.

![image](https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/0bcd3a26-f04f-4fa0-9766-871a65c70ba3)\
(figur 4. )

Det kan her ses, at vores model ofte gætter, at sandsynligheden for et godt resultat er mellem 0,4 og 0,7. Den er sjældent meget sikker på at det ender dårligt, og endnu mere sjældent at det ender godt. Det kan godt give problemer, da et usikkert resultat ikke kan aflæses, andet end at det er usikkert. Tandlægerne kan da bruge dette resultat sammen med deres andre værktøjer til at give en forudsigelse for patienten.


# Andre tilgange

I forbindelse med projektet har vi også forsøgt os med andre tilgange. Det involverer andre modeller samt dimensionalitetsreduktion i forsøg på at forbedre vores modeller.

## Lineær model
En af de første modeller vi forsøgte os med var en lineær model. Vores process kan findes under "Ekstra" i filen "Lineær regression.Rmd". I R benyttede vi funktionen lm() til at lave en lineær regression af dataet. Vi brugte da back propegation til at finde de mest relevante variable, ved at fjerne den mindst signifikante variabel og lave en ny lineær regression, indtil alle resterende variable var signifikante. Af dette kom vi fra til, at de mest relevante variable er Anteroposterior 1.1, Anteroposterior 1.2, Anteroposterior 2.2, Tooth shape/size 2 og Pan 2.2, hvor vi ser et sammenhæng med SHAP-væriderne for vores logistiske regression. Denne model er:
$$lm = 0.4430 An_{1.1} + 0.6198 An_{1.2} + 3.277 An_{2.2} - 2.581 Tss_2 - 1.929 Pan_{2.2} + 6.343$$
Det følgende plot er den egentlige værdi for Pinheiro scoren på værdierne fra vores lineære regression:
<img width="727" alt="image" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/074756f0-b463-45ce-a3b3-67c843e53e54">\
(figur 5.)\
Her ses et svagt forhold mellem de to, men det er tydeligt, at der ikke er et stærkt forhold. Særligt er der to "egenskaber" at lægge mærke til i dette plot: Vores lineære model giver mange bud omkring 10, både for lave og høje egentlige værdier. Mange af de egentlige værdier ligger omkring 0.

Til den første egenskab, kan vi bemærke, at middelværdien for modellen er nær identisk med det egentlige data, henholdsvis 8.881530 og 8.881536. Tilgengæld er variationen i vores model betydenligt lavere end for det egentlige data, henholdsvis 43.24 og 103.44. Noget af dette kan hænge sammen med den anden egenskab. I et boksplot over den egentlige værdi og lm værdierne, kan man se forskellen i spredning i pinheiro score:\
<img width="729" alt="image" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/d77ffbab-81d2-4ac8-912a-00aa82a1a25b">\
(figur 6.)\
Det kan her ses, hvordan de egentlige værdier har større spredning, og særligt hvor mange der ligger omkring 0 til 1. Boksplottet for den lineære model ligger langt mere samlet rundt om middelværdien. Da vores lineære model forsøger at mindske afstanden for hele datasættet, er det oplagt at den ligger sig rundt om middelværdien. Den gennemsnitlige afvigelse for vores model er 6.042.\
En del af dette kan også skyldes, ligesom med den logistiske reggression, at det generelt er et svagt sammenhæng mellem målinger fra 8 og 12 år og målinger fra 16 år.

Alt i alt kom vi frem til, at den lineære model ikke er god at bruge i praksis, da modellen har en tendens til at give et bud tæt ved middelværdien, hvilken den gør for lave såvel som høje værdier. Man kan derfor ikke sige meget om den endelige værdi, hvis modellen gav et resultat på mellem 7 og 10, hvilket den lader til at gøre i omkring halvdelen af tilfælde.

## Neural network.
Inden vi valgte at bruge en logistisk regression, blev muligheden for et neuralt netværk udforsket.
Vores neurale netværk består af et hidden ReLu layer og vores output layer bruger vi en softmax activation layer, som giver os en sandsynlighed for at ende i en af de to klasser. Resultateterne er dog skuffende og ender tit med at forudsige ret tilfældigt. I Jupyter notebook filen ‘Binary NN Classifier.ipynb” er det muligt at se hvordan vi har implementeret det i python, med Keras biblioteket 

Hele processen tog lang tid og endte ud i det vi fra starten lidt havde forudsagt. Vi har simpelthen ikke nok data, og for mange variable. Men det gav os en bedre forståelse for hvordan et neuralt netværk virker, men vigtigste af alt, hvornår giver det mening at bruge et neuralt netværk. Da vi endte lidt en i en blindgyde hvor modellen præcision ikke var særlig god, valgte vi at tage et skridt bagud og genoverveje hvordan vi ville takle dette projekt

## PCA og SVD.
Principal Component Analysis (PCA) er en metode, der bruges til at reducere antallet af dimensioner for data. Dette gøres ved at finde de retninger, hvor dataet spreder sig mest, og repræsentere dataet langs disse. Normalt indeholder PCA følgende trin:
1. Normalisering af variabler.
2. Beregning af kovarians matricen.
3.	Beregning af egenværdier og egenvektorer.
4.	Valg af komponenter (Vælger typisk op til at 95% varians er forklaret).
5.	Transformering af data (omdanner det oprindelige data til det nye rum defineret af de valgte komponenter).

I vores tilfælde kan 95% af variansen forklares med 14 variable - en reduktion fra 16 variable. Dog ser vi ikke en effekt på vores logistiske regression, og vælger derfor at bibeholde alle variable, da dette giver os muligheden for at benytte SHAP-værdier. 

Singular value decomposition (SVD) kan bruges, ligesom PCA, til at reducere antallet af dimensioner. Dette gøres ved at faktorisere vores data til tre matricer. Så $A = U\Sigma V^T$, hvor $A$ er en $m \times n$ matrice, $U$ er en $m \times m$ matrice bestående af de orthonormal egenvektorer fra $AA^T$, $V^T$ er en $n \times n$ matrice af $A^TA$, og $\Sigma$ er diagonalmatrice med roden af de positive egenværdier.

Med 14 komponenter er lidt over 95 procent af variansen forklaret, i vores datasæt. Vi har også her valgt ikke at benytte reduceringen.



# Konklusion
Modellen har en præcision på lige over 65%, som selvfølgelig ikke er prangende i en binær klassifikations model. Det hænger dog sammen med, at mange af datapunkterne er overvejende ens indtil sidste måling. Som diskuteret i afsnittet, om vores problemer/udfordringer.


<img width="389" alt="Dataprojekt confusion matrix" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/143393880/18d7ffe9-ac89-4359-b38c-7d7e1170b039">\
(figur 7.)

Vi har fundet de parametre, som har størst indflydelse på modellen, ved hjælp af SHAP. Det hjælper tandlægerne med forståelse og fortolkning af modellens forudsigelser.


# Referencer 

[1] Avid W. Hosmer JR, Stanly Lemeshow, Rodney X. Sturdivant: Applied Logistic Regression:  http://dl.icdst.org/pdfs/files4/7751d268eb7358d3ca5bd88968d9227a.pdf 

[2] SHAP: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
