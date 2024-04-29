# Dataproject: Prediction of craniofacial growth and occlusion for cleft lip and palate patients


## Table of contents
* [introduction](##introduktion)
* [Goals](##Mål)
* [model](#model)
* [problemer](#problemer)

# Introduktion
Læbe-ganespalte er en medfødt tilstand, som rammer omkring 1 / 500 af børn. Børn med Læbe-ganespalte gennemgår tre operationer (se figur 1). En primær operation, som er en kirurgisk lukning af læbespalten og den bløde gane. Dette sker, når patienterne er spædbørn. I en alder af enten et eller tre år lukkes den hårde gane. Når patienterne er 8 år, lukkes alveolær spalte (spalte i gummen), og bøjlebehandlingen påbegyndes. I 12-års alderen vil den endelige bøjlebehandling blive påbegyndt, og der vil eventuelt blive foretaget mindre justeringer. Tilsidst vil bøjle blive taget af, typisk i 16-års alderen.

<img width="894" alt="Skærmbillede 2024-04-03 kl  22 26 42" src="https://github.com/Christofferfuglkjaer/Dataproject/assets/120389174/ebdc42f4-f83a-4bb9-ada1-b10800b02e95">

I et internationalt studie, har man undersøgt forskellige primære operationer til sammenligning - I Danmark har man undersøgt metode a og b (i metode a lukkes den hårde gane ved 12 måneder, og i metode b, ved 36 måneder). I forbindelse med dette har man i Danmark indsamlet data på børnenes udvikling ved henholdsvis 8, 12 og 16 år.

Vi vil forsøge at lave en model, som tager værdier fra målinger af tændernes tilstand efter henholdsvis otte og tolv år, og med dette forudsige den endelige tilstand, når patienten er omkring 16 år.

Datasættet har 124 patienter med 36 kolonner, og består af tre målings-tidspunkter i hhv. aldrene 8, 12 og 16. For hver måling har vi 10 værdier for patientens tilstand, f.eks. 'Spacing, 'Transverse' og 'Crowding'. Summen af alle værdierne udgør patientens Pinheiro score. Scoren ligger mellem 0 og 52, og man ønsker en lav score.

Vi ønsker at lave en applikation gennem Streamlit, hvor tandlæger kan tilgå vores model. Gennem denne kan de indtaste værdier for de første to målinger, hvortil vores model vil give en forudsigelse for den endelige udvikling, samt hvor sikker modellen er på sin forudsigelse. Det vil også være muligt at se information omkring modellen.
Formålet med denne applikation er at patienterne kan få at vide, hvordan de kan forvente deres udvikling. Tandlægerne kan vurdere ud fra modellens forudsigelse, om en patient sandsynligvis ender med god eller dårlig udvikling. De kan også bruge modellens sikkerhed til at vurdere, om det er relevant at forberede patienten på den udvikling.

# Model 

## hvilken model bruger vi, og hvorfor (Christoffer)
Vi har valgt at bruge en logistisk regression, denne model blev valgt fordi vi ønskede at lave en binear klassificering, og med den mængde data vi har, gav det mening at bruge en supervised model, hvor labels er godt eller dårligt. 

## hvad indebærer modellen og hvordan virker den i praksis (Chrizz)

Vi bruger modellen, ved at lave en binear variable som bare er 1 hvis resultatet er godt og 0 hvis det ikke er, da vil den forudsigelse som vores model laver, være en sandsynlighed for at ligge i en af de to klasser. \
Den logistiske model er opskrevet således $$p(x) = \frac{1}{1+e^{-(x-\mu)/s}}$$ Hvor $\mu$ er det sted hvor $p(x) = \frac{1}{2}$ ## måske gennemgå det her sammen, da det er ret tung teori.\

I jupyter notebook filen 'Logistic-regression-model' bruger vi Sklearn pakken til vores model. Helt generelt så importere vi det allerede opryddet data, da standardiserer vi data og opretter vores binær kolonne. Dernæst splitter vi data op i $75 \%$ trænings data og $25\%$ test data. Nu kan vi træne vores model og bagefter bruge test data til at få hvor "god" vores model er, vi laver også en confusion matrix og en klassificerings report for bedre at kunne se, hvordan vores modellen gætter. Da vi havde store svingninger i vores models præcision, valgte vi at bruge en bootstrap tilgang, hvor vi kører modellen 25 gange og tager middelværdien af forudsigelse, præcision og SHAP værdierne. Det har gjort at vores model er mere stabil og giver en mere ensartet forudsigelse. 
 
## SHAP værdier og hvad bruger vi dem til. (Oswald)
SHAP (SHapley Additive exPlanations) er en metode, der kan bruges til at fortolke/forklare Machine Mearning modellers forudsigelser. Mere specifikt, kan man se hver parameters effekt på en forudsigelse.\
Når man arbejder med SHAP-værdier, er det vigtigt at notere sig, at de ikke kan bruges til at forklare kausalitet. Siger udelukkende noget om, hvordan modellen er kommet frem til en forudsigelse/resultat.\
\
Vi bruger to plots fra pakken 'shap', til forklare modellens forudsigelser. De kan findes under 'Extra Information' i streamlit appen. Her kan man se, at de fleste gange modellen bliver kørt, vil Antereoposterior 1.1 være den parameter med størst effekt. På det andet plot ses, hvor uforudsiglig problemstillingen egentlig er - der er ikke stor sammenhæng mellem parameter-værdien og shap-værdien. (Optimalt ville røde og blå punkter være adskilt).\

https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability

## problemer 

# andre tilgange ()
## neural network (Christoffer)
Inden vi valgte at bruge en logistisk regression, blev muligheden for et neuralt netværk udforsket, dette tog lang tid og endte ud i det vi  fra starten lidt havde forudsagt. Vi har simpelthen ikke nok data, og for mange variable. Men det gav os en bedre forståelse for hvordan et neuralt netværk virker, men vigtigste af alt, hvornår giver det mening at bruge et neuralt netværk. Da vi endte lidt en i en blindgyde hvor modellen præcision ikke var særlig god, valgte vi at tage et skridt bagud og genoverveje hvordan vi ville takle dette projekt. 

## verdens bedste LM (Malthe)
## PCA og SVD. (Oswald)
Principal Component Analysis er en metode, der bruges til at reducere antallet af dimensioner for data. Dette gøres ved at finde de retninger, hvor dataet spreder sig mest, og repræsentere dataet langs disse. Normalt indeholder PCA følgende trin:\
-	Normalisering af variabler – trækker gennemsnittet fra.\
-	Beregning af kovarians matricen.\
-	Beregning af egenværdier og egenvektorer.\
-	Valg af komponenter (Vælger typisk op til at 95% varians er forklaret).\
-	Transformering af data (omdanner det oprindelige data til det nye rum defineret af de valgte komponenter).\

I vores tilfælde kan 95% af variansen forklares med 12 variable (starter på 18). Dog ser vi ikke en effekt på vores regression, og vælger derfor at bibeholde alle variable. Da dette også giver os muligheden for at benytte SHAP-værdier. 


# opnåede vi de mål? (team combo) 

## lave upload tamtam 
## Streamlit app (Christoffer)
Da vi gerne vil have at tandlægerne kan bruge den model vi har lavet som et værktøj, ønskede vi fra starten at gøre det så nemt som muligt for dem at indtaste nye tal og så få en forudsigelse tilbage. Her har vi brugt en python pakke som hedder Streamlit, som ligeledes hoster hjemmesiden i skyen for os. Det betyder at vi laver et slags dashboard, det gør det nemt for tandlægerne at bruge vores model uden at skulle installere python eller overhovedet forstå det tekniske bag vores model og forudsigelse. (Ligeledes er det muligt for dem at oploade mere data, som vores model så kan bruge i fremad.)\
Det har været vanskeligt og meget tid er langt i hjemmesiden, da vi skulle lære en hvordan Streamlit fungerer. Vi opfordre at man går ind og kigger på hjemmesiden. 
link til hjemmeside:  https://cleft-lip-app-r4y7280urvh.streamlit.app
