# Forudsigelse af kraniofacial vækst og tandstilling for læbe-ganespaltepatienter


## Indholdsfortegnelse
* [Introduktion](#introduktion)
* [Model](#model)
* [SHAP værdier](#shap-værdier)
* [Udfordringer](#udfordringer)
* [Andre tilgange](#andre-tilgange)
* [Resultater](#resultater)
* [Referencer](#referencer)

# Introduktion
Læbe-ganespalte er en medfødt tilstand, som rammer omkring hvert 500. barn. Børn med læbe-ganespalte gennemgår tre operationer (se figur 1). En primær operation, som er en kirurgisk lukning af læbespalten og den bløde gane. Dette sker, når patienterne er spædbørn. Den sekundære operation sker i en alder af enten et eller tre år, hvor den hårde gane lukkes. Når patienterne er 8 år får de den tredje operation, hvor man lukker den alveolære spalte (spalten i gummen), og første bøjlebehandling påbegyndes. I 12-års alderen vil den endelige bøjlebehandling blive påbegyndt, og der vil eventuelt blive foretaget mindre justeringer. Til sidst vil patientens bøjle blive taget af samt endelige justeringer, typisk i 16-års alderen.

<img width="894" alt="Skærmbillede 2024-04-03 kl  22 26 42" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/ebdc42f4-f83a-4bb9-ada1-b10800b02e95">\
(figur.1: forløb for ganespaltepatienter)\
I et internationalt studie, har man sammenlignet forskellige sekundære operationer i en population af patienter med læbe ganesplatning. I forbindelse med dette har man i Danmark indsamlet data på børnenes udvikling ved henholdsvis 8, 12 og 16 år.

 Datasættet har 123 patienter med 36 kolonner, og består af tre målings-tidspunkter i henholdsvis aldrene 8, 12 og 16. For hver måling har vi 10 værdier for patientens tilstand, f.eks. 'Spacing', 'Transverse' og 'Crowding'. Summen af alle værdierne i en måling udgør patientens Pinheiro score. Scoren ligger mellem 0 og 52, og man ønsker en lav score.

Vores model tager værdier fra målinger af tændernes tilstand efter henholdsvis 8 og 12 år, og vil prædiktere den endelige tilstand når patienten er 16 år, baseret på tidligere datapunkter. Dette bliver gjort ved at lave en binær variabel, som er god, hvis den endelige Pinheiro score er 5 eller under, hvilket vi klassificerer som 0, og alt over 5 er et dårligt resultat, hvilket vi klassificerer som 1. Vi har valgt 5 som grænsen i samarbejde med tandlægerne, da det er medianen for endelige Pinheiro-score og stemmer overens med et lægefagligt godt resultat. 


For at tandlægerne kan tilgå og bruge modellen i praksis, er der udviklet en app igennem Streamlit. Gennem appen kan tandlægerene  indtaste værdier for de første to målinger, hvortil vores model vil give en prædiktion for den endelige udvikling, samt hvor sikker modellen er på sin forudsigelse. Appen skal bruges som et værktøj af tandlægerne til at understøtte deres faglige intuition og de andre midler de benytter.


# Model 

## Logistisk regression
Vi har valgt at bruge en multiple logistisk regression, da vi ønsker at lave en binær klassificering. Det giver derfor mening at bruge en supervised model, hvor labels er givet ud fra vores binære variabel.
Vi vil gennemgå teorien bag den multiple logistisk regression, og derefter hvordan vi implementerer og fitter vores model i python.

## Teori

Givet vi har n datapunkter, som er I.I.D og er angivet på formen $X = [x_1,x_2,...,x_n]$. Vi definerer nu et $Y$, som er en "dummy variabel", der er 0 eller 1. Lad nu $\pi(x)$ være en betinget sandsynlighed, som er $P(Y=1|x) = \pi(x)$, og $P(Y = 0|x) = 1-\pi(x)$. Da er logit af vores multipel regression givet som: 
$$g(x) =\ln\left(\frac{\pi(X)}{1-\pi(X)}\right) = \beta_0+\beta_1x_1+...+b_n x_n$$


Vi kan nu opskrive vores multipel logisitiske regression, den har formen af en logistisk sigmoid:

$$\pi(x)=\frac{e^{g(x)}}{1+e^{g(x)}}$$

Vi kan nu benytte Bernoulli fordelingen til at opstille vores likelihood funktion.

$$\pi(x_i)^{y_i}(1-\pi(x_i))^{1-y}$$
Vi ved at alle observationer er uafhængige, da det er målinger fra forskellige patienter. Da er vores likelihood funktion et samlet produkt af ovenstående udtryk.

$$l(\beta')=\prod^n_{i=1} \pi(x_i)^{y_i} (1-\pi(x_1))^{1-y}$$
hvor $\beta' = [\beta_0,...,\beta_m]$

Princippet bag maksimum likelihood funktionen er at estimere den værdi for hver $\beta_0...\beta_m$ i $\beta'$ , som maksimerer udtrykket. For at gøre det nemmere at estimere $\beta'$, benytter vi Log-likelihood methoden og omskriver  $l(\beta')$ til 

$$L(\beta')=\ln(l(\beta')) = \sum^n_{i=1} y_i \ln(\pi(x_i))+(1-y_i)\ln(1-\pi(x_i))$$

Nu differencierer vi $L(\beta')$ med respekt til $\beta'$ for at finde de værdier, som maksimerer vores udtryk.
$$\hat{\beta'}= \frac{\partial L(\beta')}{\partial \beta_'} = \sum^n_{i=1} y_i x_{im} - x_{im} \pi(x_i) = 0$$
Nu har vi fundet vores maksimum likelihood estimater, som vi beskriver ved $\hat{\beta'} = [\hat{\beta_0},...,\hat{\beta_m}]$$

Nu hvis vi gerne vil lave en forudsigelse med vores model, benytter vi $\hat{\beta_0...\hat{\beta_m}}$ og indsætter dem i 
$$\pi(x) = \frac{e^{\hat{\beta_0}+\hat{\beta_1}x_1+...+\hat{\beta_m}x_m}}{1+e^{\hat{\beta_0}+\hat{\beta_1}x_1+...+\hat{\beta_m}x_m}}$$


Alt teorien er fundet i [1] s.6-9 og s.31-34.


I Jupyter notebook 'Logistisk-regression.ipynb' bruger vi $\texttt{Sklearn}$ til konstruere og fitte vores model. Helt generelt, så importerer vi det behandlede data, som vi standardiserer med en $\textit{MinMaxScaler}$, hvilket betyder at datapunkter bliver skaleret til at være mellem 0 og 1. Da kan vi oprette vores binær target kolonne. Dernæst splitter vi data op i 75 procent trænings data og 25 procent test data. Nu kan vi fitte vores model og bruge test data til analysere hvor præcis vores model er. Da vi havde store svingninger i vores models præcision, valgte vi at bruge en Bootstrap tilgang, hvor vi kører modellen 25 gange, med tilbagelægning, og tager middelværdien af forudsigelserne, præcision og SHAP-værdierne. Det har gjort at vores model er mere stabil og giver en mere ensartet forudsigelse. 

## Streamlit app 
En del af projekt var at finde en måde hvormed tandlægerne kunne benytte vores model, uden at have kendskab til Python eller den bagved liggende matematik. Det endte ud i en app, hvor tandlægerne kan bruge den model vi har lavet som et værktøj. Det er gjort nemt for dem at indtaste målinger og få en forudsigelse tilbage. For at lave appen har vi brugt Python pakken 'Streamlit', som producerer en cloud-baseret hjemmeside.

Appen bruger vores bootstrap logistisk regression som i 'logistisk regression.ipynb' filen, og viser præcision, spredning og SHAP-værdier.

Det betyder at vi laver et slags dashboard, der gør det nemt for tandlægerne at bruge vores model uden at skulle installere Python eller overhovedet forstå det tekniske bag vores model og forudsigelse. 
 
Link til appen:  https://cleft-lip-app-r4y7280urvh.streamlit.app

 
## SHAP værdier
SHAP (Shapley Additive exPlanations) er en metode, der kan bruges på machine learning modeller, til at se hver parameters effekt på en forudsigelse.\
Når man arbejder med SHAP-værdier, er det vigtigt at notere sig, at de ikke kan bruges til at forklare kausalitet. De siger udelukkende noget om, hvilke parametre modellen har brugt til at komme frem til en forudsigelse.\
\
Vi bruger et plot fra pakken SHAP, til forklare modellens forudsigelser. Det er muligt at se hvordan SHAP værdier bliver lavet i 'logistisk regression.ipynb'. Her kan man se, at de fleste gange modellen bliver kørt, vil Antereoposterior 2.1 være den parameter med størst effekt.

![image](https://github.com/Christofferfuglkjaer/Dataproject/assets/118052934/026fdeb7-9239-4f76-a84a-c99aae7f6f91)\
(figur 2. SHAP plot)

## Udfordringer 

PDen største udfordring med dette projekt er fokuseret omkring det data gjort tilgængeligt. Vi har været begrænset af størrelsen af vores data, der indeholder 123 patienter, som betyder, at vi skal være opmærksomme på overfitting. Yderlige har der været datapunkter med manglende værdier, så vi ender med 116 datapunkter. Datoerne for målingerne er tilgængelige for patienter fra hovedstaden, men manglende for patienter fra Aarhus, hvilket tvinger os til ikke at bruge dem. De kunne have haft en effekt, da patientens alder ved sidste måling varierer en del. Dette skyldes, at patienten skal være helt færdig med behandlingen før de får taget de sidste mål.

Begrænsningen af et lille datasæt har vi forsøgt at undgå i vores logistiske regressionsmodel ved at bruge Bootstrapping. Her vil vi lave modellen på forskellige dele af datasættet, hvor den gennemsnitlige model da gerne skulle have mindre overfitting.


Derudover har datasættet ikke en stærk sammenhæng mellem målingerne fra 8 og 12 år og målingerne fra 16 år. Særligt har vi set, at patienter, som har en høj score i de to første målinger, kan ende med både gode og dårlige resultater i sidste måling, som kan ses i figuren under.

<img width="730" alt="image" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/ac63ddf7-863b-44bc-9182-4893c11ec845">\
(figur 3. Målinger fra 8-12-16 år)

For patienter med en Pinheiro score over 20 i målingen ved 12 år ender omkring 37\% af dem med et godt resultat. Det giver en del usikkerhed i vores model, som derfor har en tendens til at gætte imellem. I denne figur har vi kørt modellen 10.000 gange med et udsnit af datasættet som træningsdata, og observeret hvor sikker den er på, at hver testpatient ender med et godt resultat.

![image](https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/0bcd3a26-f04f-4fa0-9766-871a65c70ba3)\
(figur 4. )

Det kan her ses, at vores model ofte gætter, at sandsynligheden for et godt resultat er mellem 0.4 og 0.7. Den er sjældent meget sikker på at det ender dårligt, og endnu mere sjældent sikker på at det ender godt. Et usikkert resultat kan være svært for tandlægerne at bruge i deres forudsigelse.


# Andre tilgange

I forbindelse med projektet har vi også forsøgt os med andre tilgange. Det involverer andre modeller samt dimensionalitetsreduktion i forsøg på at forbedre vores modeller.

## Lineær model
Den første model vi forsøgte os med var en lineær model. Koden til dette kan findes under 'Andre tilgange' i filen 'Lineær regression.Rmd'. I R benyttede vi funktionen $\textit{lm()}$ til at lave en lineær regression af dataet.

Vi brugte da backwards selection til at finde de mest relevante variable. Dette sker ved at fjerne den mindst signifikante variabel, lave en ny lineær regression og gentage, indtil alle resterende variable var signifikante. Af dette kom vi fra til, at de mest relevante variable er Anteroposterior 1.1, Anteroposterior 1.2, Anteroposterior 2.2, Tooth shape/size 2 og Pan 2.2, som stemmer overens med SHAP-værdierne for vores multiple logistiske regression. Herunder ses den lineære model:
$$lm = 0.4430 An_{1.1} + 0.6198 An_{1.2} + 3.277 An_{2.2} - 2.581 Tss_2 - 1.929 Pan_{2.2} + 6.343$$
Denne model har en forklaringsgrad på 0.4, og den gennemsnitlige afvigelse fra den egentlige værdi for Pinheiro scoren er 6.042.

I figur 5 herunder er den egentlige værdi for  Pinheiro scoren af x-aksen imod værdierne fra vores lineære regression af y-aksen:
<img width="727" alt="image" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/074756f0-b463-45ce-a3b3-67c843e53e54">\
(figur 5. Linenær model scatterplot)\
Af dette plot kan vi se begrænsningerne i vores lineære model. Mange af de egentlige værdier for Pinheiro scoren ender i nul, mens den lineære model giver værdier mellem nul og 10 for disse patienter. På samme vis ser vi en gruppe af egentlige værdier omkring 30, som modellen giver en værdi omkring 10.
Generelt ser vi et svagt sammenhæng, idet værdierne af den lineære model stiger sammen med de egentlige værdier. Men outputtet fra modellen kan variere meget fra den egentlige værdi.

Vi bemærker også, at middelværdien for modellen er nær identisk med det oprindelige data, henholdsvis 8.881530 og 8.881536. Til gengæld er variationen i vores model markant lavere end for det oprindelige data, henholdsvis 43.24 og 103.44. I et boksplot over den egentlige værdi og lm værdierne kan man se forskellen i spredning for Pinheiro scoren:\
<img width="729" alt="image" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/d77ffbab-81d2-4ac8-912a-00aa82a1a25b">\
(figur 6.Lineær model boxplot)\
Det kan her ses, hvordan de egentlige værdier har større spredning, og særligt hvor mange der ligger omkring 0 til 1. Boksplottet for den lineære model ligger langt mere samlet rundt om middelværdien.\
Ligesom med den multiple logistiske regression konkluderer vi, at der generelt er et svagt sammenhæng mellem målinger fra 8 og 12 år og målinger fra 16 år.

Alt i alt kom vi frem til, at den lineære model ikke er god at bruge i praksis, da modellen har en tendens til at give et bud tæt ved middelværdien, hvilket den gør for lave såvel som høje værdier. Man kan derfor ikke sige meget om patientens endelige Pinheiro score, hvis modellen giver et resultat på mellem 7 og 10.

## Neural network
Inden vi valgte at bruge en logistisk regression, blev muligheden for et neuralt netværk udforsket.
Vores neurale netværk består af et hidden ReLu layer og til vores output layer bruger vi et softmax activation layer, som giver os en sandsynlighed for at ende i en af de to klasser. Resultateterne ender dog tit med at forudsige ret tilfældigt. I Jupyter notebook filen 'Binary NN Classifier.ipynb' er det muligt at se hvordan vi har implementeret det i Python, med $\texttt{Keras}$ biblioteket 

Hele processen endte ud i det, vi fra starten lidt havde frygtet. Vi har simpelthen ikke nok data, og for mange variable. Da vi endte lidt en i en blindgyde, hvor modellens præcision ikke blev forbedret, valgte vi at tage et skridt bagud og genoverveje hvordan vi ville takle dette projekt, hvor vi endte med logistisk regression.

## PCA og SVD
Principal Component Analysis (PCA) er en metode, der bruges til at reducere antallet af dimensioner for data. Dette gøres ved at finde de akser, hvor dataet har størst variation, og projicerer dataet langs disse. Normalt indeholder PCA følgende trin:
1. Normalisering af variabler.
2. Beregning af kovarians matricen.
3.	Beregning af egenværdier og egenvektorer.
4.	Valg af komponenter (Vælger typisk op til at 95% varians er forklaret).
5.	Transformering af data (omdanner det oprindelige data til det nye rum defineret af de valgte komponenter).

I vores tilfælde kan 95% af variansen forklares med 14 variable - en reduktion fra 16 variable. Dog ser vi ikke en effekt på vores logistiske regression, og vælger derfor at bibeholde alle variable, da dette giver os muligheden for at benytte SHAP-værdier. 

Singular value decomposition (SVD) kan bruges, ligesom PCA, til at reducere antallet af dimensioner. Dette gøres ved at faktorisere vores data til tre matricer. Så $A = U\Sigma V^T$, hvor $A$ er en $m \times n$ matrice, $U$ er en $m \times m$ matrice bestående af de orthonormale egenvektorer fra $AA^T$, $V^T$ er en $n \times n$ matrice af $A^TA$, og $\Sigma$ er diagonalmatrice med roden af de positive egenværdier.

Med 14 komponenter er lidt over 95 procent af variansen forklaret i vores datasæt. Vi har også her valgt ikke at benytte reduceringen, da vi vil miste fortolkningsevnen af SHAP-værdierne.

## Andre statiske modeller

Vi undersøgte også andre klassificeringsmodeller, som $\texttt{RandomForrestClassifier}$, $\texttt{DecisionTreeClassifier}$ og $\texttt{KNeighborsClassifier}$. Som man kan se i 'Andre statistiske modeller.ipynb' så får vi den samme præcision som Logistisk regression, og da logistisk regression var hurtigere om at fitte, og vi ønskede at appen var effektiv valgte vi at beholde vores multiple logistiske regression. 

# Resultater
Formålet med projektet var at forsøge at prædiktere om den endelige Pinheiro-score ville ende i den gode eller dårlige kategori. Det er til en vis grad lykkedes. Modellen har en præcision på lige over 65%, som selvfølgelig ikke er prangende i en binær klassifikations model. Det hænger dog sammen med, at mange af datapunkterne er overvejende ens indtil sidste måling. Som diskuteret i afsnittet, om vores udfordringer. Dog kommer denne usikkerhed også til udtryk i modellen, da den både viser sandsynligheden for at ende i 0 som er et godt resultat, men også for at ende i 1 som er et dårligt resultat. Dvs. at tandlægerne kan se, når prædiktion kan kategoriseres som ugyldig eller for usikker.

Vi bruger her en confusion matrix til at visualisere modellens præcision, hvor nederste venstre boks er antallet af falske positive og øverste højre boks er falske negative.


<img width="389" alt="Dataprojekt confusion matrix" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/143393880/18d7ffe9-ac89-4359-b38c-7d7e1170b039">\
(figur 7. Confusion matrix)

Vi ville også undersøge, hvilke parametre, der mest indflydelse på det endelige resultat. Det blev gjort ved hjælp af SHAP-værdier, som viser at de tre parametre, der har størst effekt på modellens prædiktion, er Anteroposterior 2.1, Pan 2 og Anteroposterior 1.1. Ligeledes kan man se hvor meget de andre parametere påvirker modellens prædiktion. 

Tandlægerne kan da bruge modellen som et værktøj i deres forudsigelse af en patients udvikling til at understøtte deres faglige intuition.


# Referencer 

[1] Avid W. Hosmer JR, Stanly Lemeshow, Rodney X. Sturdivant: Applied Logistic Regression:  http://dl.icdst.org/pdfs/files4/7751d268eb7358d3ca5bd88968d9227a.pdf 

[2] SHAP: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
