# AML Images Labeling VS Labeling In Opensource

### **Opis**

Projekt zakłada przegląd narzędzi umożliwiających oznaczanie obrazów. Szczególnej uwadze ma zostać poddany Azure Machine Learning Data Labeling. Omówione poniżej narzędzia, mają za zadanie ułatwienie przygotowania zbiorów danych, po przez ich oznaczenie. Zbiór taki może posłużyć następnie do dalszej analizy, zostać wykorzystany w kolejnym narzędziu lub może posłużyć jako zbiór danych dla uczenia sieci.

### Zespół - grupa nr 5 - *gamma*

- [Mateusz Mizio ](https://github.com/miziom ) - MM

- [Jarosław Królik](https://github.com/j-krolik) - JK

### Repozytorium

Repozytorium GitHub - [LINK](https://github.com/miziom/AI-Azure-Project-Image-Labeling)

### Wybrany stos technologiczny

***Serwis Azure***

- Azure Machine Learning - Data Labeling
  - Image Classification Multi-class
  - Image Classification Multi-label
  - Object Identyfication (Bounding Box)
  - Instance Segmentation (Polygon) (Preview)

***Inne technologie:*** 

- Cvat
- Label Studio
- Labelbox
- Coco Annotator
- Awesome Data Labeling
- Yolo_label

***Pochodzenie zbiorów danych***

- [ZBIÓR KWAITÓW](https://www.kaggle.com/alxmamaev/flowers-recognition)
- [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html)

### Kalendarz

|  Nr   |                          Działania                           |          Data           |    Kto?    | Check |
| :---: | :----------------------------------------------------------: | :---------------------: | :--------: | :---: |
|   1   |   Uzupełnienie stosu technologicznego - co będziemy badać?   |       do 13.12.20       |   MM, JK   |   X   |
|   2   | Analiza *Azure Machine Learning - Data Labeling* (uszczególnienie) |       do 20.12.20       |   MM, JK   |   X   |
|   3   |       Przygotowanie zbioru danych potrzebnego do pracy       |       do 22.12.20       |   MM, JK   |   X   |
| **4** |                    **Przerwa Świąteczna**                    | **23.12.20 - 01.01.21** | **MM, JK** | **X** |
|   5   |               Analiza *Azure Machine Learning*               |       do 10.01.20       |     MM     |   X   |
|   6   |                                                              |                         |            |       |
|   7   |                                                              |                         |            |       |
|   8   |                                                              |                         |            |       |
|   9   |                                                              |                         |            |       |
|  10   |                                                              |                         |            |       |

### Opis funkcjonalności i schematy działania

1. ##### Azure Machine Learning - Data Labeling

   Narzędzie to jest miejscem do tworzenia i monitorowania projektów etykietowania oraz zarządzania nimi po przez:

   - zarządzaniem danymi, etykietami, członkami zespołu
   - śledzenie postępów w projekcie
   - uruchamianie i zatrzymywanie projektu
   - przeglądanie oznaczonych danych
   - eksport oznaczonych danych

   Niezależnie od typu projektu (4 możliwości, opisane w dalszej części), należy spełnić pewne wymagania: 

   - <u>wprowadzenie zbioru danych</u> 

     Podajemy jego nazwę oraz możemy zamieścić jego opis. Wcześniej jednak musimy zdecydować jakiego typu będzie to zbiór. Zbiór może liczyć ***maksymalnie 500 000 obrazów***. Mamy dwie możliwości:

     - na podstawie plików lokalnych

       W ten sposób możemy prosto dodać pliki lokalne do utworzonego wcześniej lub na bieżąco Azure Blob Storage. Tworząc projekt AML mamy utworzony domyślny *workspaceblobstore*, do którego możemy dodać obrazy.

     - na podstawie magazynów danych - datastore (nie są wspierane magazyny danych SQL):

       - Azure Blob Storage
        - Azure File Share
       - Azure Data Lake Storage Gen1
        - Azure Data Lake Storage Gen2

   - <u>określenie "Incremental Refresh"</u> - *opcjonalne*

     Warto używać tej funkcji, jeżeli będziemy chcieli uzupełniać zbiór danych. Obrazy możemy dodawać za pomocą [Azure Storage Explorer](https://azure.microsoft.com/features/storage-explorer/). Po dodaniu obrazów do zbioru, zobaczymy, że dane są zamieszczone, jednak utworzony projekt *AML Data Labeling* nie będzie widział zamieszczonych nowych danych. Dlaczego? Dokumentacja przedstawia, że po włączeniu omawianej funkcji dodane dane będą pobierane do projektu raz dziennie. W praktyce dane aktualizowane są po 3-6h po dodaniu do zbioru danych.

   - <u>określenie etykiet dla poszczególnych klas</u>		

     Należy dodać etykiety, aby móc skategoryzować dane. Nazwy etykiet powinny być jednoznaczne, aby następnie osoby, które będą tagować zdjęcia nie miały wątpliwości co oznaczają. Należy dodać minimum 2 etykiety, aby stworzyć projekt. Etykiety można edytować po stworzeniu projektu, jednak należy pamiętać, że projekt musi wtedy być zatrzymany.

   - <u>dodanie instrukcji</u> - *opcjonalne*

     W celu poinstruowania pozostałych pracowników projektu, można dodać instrukcje jak oznaczać obrazy. Można zrealizować to na dwa sposoby:
     - opisać w polu tekstowym

     - podać adres URL do zewnętrznej instrukcji - przydatne dal bardziej skomplikowanych instrukcji, albo w przypadku gdy taki zestaw zalecań został stworzony już wcześniej i jest opublikowany

     Instrukcje należy sformułować tak, aby rozwiewała wszelkie wątpliwości. Dobrze określić  etykietującym co powinni zrobić, jeśli żadna etykieta nie wydaje się odpowiednia, lub co powinni zrobić jeżeli kilka etykiet wydaje się być odpowiednimi. Co powinni zrobić, jeśli interesujący obiekt zostanie przycięty przez krawędź obrazu? Co powinni zrobić po przesłaniu etykiety, jeśli uważają, że popełnili błąd? Jeżeli mają definiować obwiednie, dobrze poinstruować w jaki sposób powinni ją tworzyć. Czy powinno znajdować się w całości wewnątrz obiektu, czy też na zewnątrz? Czy powinien być przycięty tak blisko, jak to możliwe, czy też dopuszczalny jest pewien prześwit? Jak oznaczyć przedmiot, który jest częściowo zakryty innym przedmiotem?

     Im lepiej zostanie stworzona instrukcja, tym mniej będzie wątpliwości, a co za tym idzie powstanie lepiej oznaczony zbiór danych.

   - <u>etykietowanie wspomagane przez **Machine Learning**</u> - *opcjonalne*

     Etykietowanie może być wspomagane przez Machine Learning. Umożliwia wyzwalanie modeli automatycznego uczenia maszynowego w celu przyspieszenia zadania etykietowania. Przy starcie projektu należy oznaczyć wiele obrazów. **Nie ma określonego progu**, po którym ML zostaje wyzwolony. Wszystko zależy od tego na jakim zbiorze danych tworzony jest projekt oraz od tego na ile poprawnie zostały oznaczone wcześniejsze obrazy.

     Z przeprowadzonych testów na zbiorach 1000, 2000 oraz ok 3000 obrazów zauważono, że próg przy poprawnym oraz dokładnym (w przypadku obwiedni) etykietowaniu wynosi między **300**, a **500** obrazów. Jednak pierwsze treningi wykonywane są już po ok 75-100 oznaczeniach. Jednak początkowe trening są wykonywane w ramach dostosowania do wstępnie wytrenowanego modelu. Etykietowanie wspomagane przez ML wykorzystuje **Transfer Learning**. Oznacza to, że wykorzystuje wstępnie wytrenowany model do szybkiego rozpoczęcia procesu szkolenia. Im większe podobieństwo klas użytych w projekcie do klas zdefiniowanych w modelu wstępnie wytrenowanym, tym szybsze rozpoczęcie wspomagania projektu przez ML.

     Ponieważ ostateczne etykiety nadal opierają się na danych wejściowych od osób etykietujących, technologia ta jest nazywana ***human in the loop labeling***.

     Aby móc skorzystać z ML należy użyć Maszyny Wirtualnej. W zależności od subskrypcji mamy dostęp do różnych klastrów. Maszyny możemy stworzyć podczas tworzenia projektu, lub wybrać wcześniej utworzone. 

     Platforma AML udostępnia nam 2 typy maszyn wirtualnych, jeżeli chodzi o ich priorytet:

     - dedykowane - działają płynnie, bez przerwań

     - o niskim priorytecie - tańsze, ale nie gwarantują węzłów obliczeniowych, może okazać się, że praca zostanie przerwana lub rozpoczęta z dużym opóźnieniem

     Można korzystać tylko z maszyn wirtualnych z obsługą **GPU**, co korzystnie wpływa na szybkość obliczeń, szczególnie kiedy chodzi o przetwarzanie obrazów. 

     Oprócz wyboru priorytetu, istnieje możliwość wyboru wielkości tworzonej maszyny. Obsługiwane rozmiary maszyn wirtualnych mogą być ograniczone w zależności od regionu. Microsoft publikuje listę maszyn oraz ich dostępności do poszczególnych regionów - [LINK](https://azure.microsoft.com/en-us/global-infrastructure/services/?products=virtual-machines&regions=all). 

     W ramach używanej licencji studenckiej całkowity dostępny limit wynosi **6 rdzeni**. 

     W przypadku wielkości można wybierać pomiędzy dwiema opcjami:

     - polecanymi

       Poniżej przedstawiono dostępne polecane VM dla rekomendowanych rozmiarów oraz pozostałych. Jak widać w zależności od możliwości Maszyny zmienia się jej cena pracy za godzinę. 

       ![](AML - screens\2_vm_dedicated_recommended.PNG)

       ![](AML - screens\4_vm_dedicated_all.PNG)

     - wszystkimi opcjami

       Poniżej przedstawiono dostępne polecane VM dla rekomendowanych rozmiarów oraz pozostałych. Tu również możliwości maszyn przekładają się na ceny jednak są znacznie tańsze względem *polecanych*. Jeżeli jednak chcemy zaoszczędzić pieniądze lub nie zależy nam na ciągłości pracy Maszyny Wirtualnej, jest to odpowiednia opcja 

       ![](AML - screens\3_vm_low_recommended.PNG)

       ![](AML - screens/5_vm_low_all.PNG)

     Ze względu na ustalony limit 6 rdzeni, jak można zauważyć powyżej, wiele maszyn nie może zostać nam udostępnionych. Limity wprowadzane są ze względu na to, aby nie przekroczyć budżetu oraz przestrzegać ograniczeń pojemności platformy Azure. Ze względu na to, że konto Azure na subskrypcji studenta posiada $100 kredytów,  wiele VM zostało zablokowanych, ponieważ użycie ich spowodowałoby szybkie zużycie środków.

     W projektach mogą wystąpić **3 etapy** pracy:

     - **etap ręcznego oznaczanie**

       Początkowo należy ręcznie oznaczyć około 100 obrazów, aby pracę rozpoczął Machnie Learning. Po około 350-500 oznaczonych obrazach dostępne są kolejne etapy.

       ![](AML - screens\gifs\manual.gif)

     - **etap klastrowania** - nie występuje w projektach wykrywania obiektów 

       Po odpowiedniej ilości wyetykietowanych obrazów, model zaczyna grupować podobne obrazy. Sprawia to, że pogrupowane obrazy prezentowane są osobom etykietującym na ekranie na wybranych przez nich siatkach 4, 6 lub 9 obrazów. Umożliwia to znaczne przyspieszenie pracy, ponieważ oznaczający nie muszą poświęcać dodatkowego czasu na za wybieranie pomiędzy klasami. 

       Po wstępnym wytrenowaniu modelu na danych oznaczonych ręcznie, model jest ograniczany do ostatniej w pełni połączonej warstwy. Obrazy bez etykiety są następnie przepuszczane przez wycięty/ograniczony model w procesie znanym jako cechowanie (featurization). To umieszcza każdy obraz w wielowymiarowej przestrzeni zdefiniowanej przez tę warstwę modelu. Obrazy, które są najbliższymi sąsiadami w przestrzeni, są używane do zadań grupowania.

       ![](AML - screens\gifs\clustered.gif)

     - **etap wstępnego oznaczania**

       Po przesłaniu wystarczającej liczby etykiet obrazów do przewidywania znaczników obrazów używany jest model klasyfikacji. Liczba ta jest indywidualna dla każdego projektu i zmienia się wraz z jego postępem. W tym etapie osoby etykietujące widzą już oznaczone zdjęcia, a w przypadku projektu wykrywania obiektów widzą przewidywane pola. Dzięki temu ich praca zawęża się do sprawdzania, czy algorytm oznaczył dobrze zdjęcia. Mogą nanosić wszelkie poprawki w przypadku błędnych oznaczeń lub jeżeli nie zostały oznaczone wszystkie obiekty, które powinny zostać oznaczone.

       Po przeszkoleniu modelu uczenia maszynowego na danych oznaczonych ręcznie, model jest oceniany na zestawie testowym ręcznie oznaczonych obrazów w celu określenia jego dokładności przy różnych progach ufności. Ten proces oceny służy do określenia progu ufności, powyżej którego model jest wystarczająco dokładny, aby pokazać wstępne etykiety. Model jest następnie oceniany na podstawie nieznakowanych danych. Obrazy z przewidywaniami bardziej wiarygodnymi niż ten próg są używane do wstępnego etykietowania.

       ![](AML - screens\gifs\prelabeled.gif)

     Kiedy etykietowanie wspomagane ML jest włączone, na ekranie podsumowującym projekt pokazywany jest szereg informacji. Mały pasek postępu pokazuje, kiedy nastąpi następny trening. Możemy podejrzeć ile zdjęć zostało przypisanych do danych etapów, jak rozkładają się poszczególne etykiety oraz ile oznaczeń wykonali poszczególni tagujący.

     Poniżej pokazane są 3 sekcje, które pokazują poszczególne eksperymenty:

     - *training*

       Odpowiada za naukę modelu przewidywania etykiet. Wykonywany jest wielokrotnie podczas trwania projektu. Dla każdego treningu wyznaczana jest dokładność jak i precyzja. Proces ten jest o tyle fascynujący, że reagują na bieżące postępy projektu. Jeżeli jesteśmy na etapie *wstępnego oznaczania* to nawet jeżeli nic nie będziemy zmieniać tylko potwierdzać przypuszczenia i oznaczenia modelu, to zatwierdzone przez nas wyniki będą  wykorzystane do kolejnych treningów. Wtedy występuje sytuacja, że model uczy się w pełni sam i poprawia swoje wyniki. 

       ![](AML - screens\7_traning_runs.PNG)

     - *validation*

       Określa, czy predykcja tego modelu będzie używana do wstępnego etykietowania pozycji. Podczas przeprowadzanych wielu próbach na różnych rodzajach projektów ani razu eksperyment nie został oznaczony jako wykonany, jednak przejście do kolejnych eksperymentów może sugerować, że jest to błąd wyświetlania informacji.

     - *inference*

       Odpowiada za przebieg prognozy dla nowych pozycji.

     - *featurization* - tylko dla projektów klasyfikacji obrazów

       Odpowiada za elementy klastrów. Umieszczane są obrazy w przestrzeni zdefiniowanej przez warstwę obciętego modelu. Obrazy, które są najbliższymi sąsiadami w przestrzeni, są używane do zadań grupowania.

     ![](AML - screens\6_dashboard.PNG)

     

     Narzędzie umożliwia utworzenie **4 rodzajów projektów**:

     ![](AML - screens\1_task_type.PNG)

     - **Image Classification Multi-class**

       Projekt, który umożliwia oznaczenie obrazu tylko jedną klasą z zestawu klas. 
       Obejmuje wszystkie omówione wcześniej 3 etapy.

       ![](AML - screens\16_work_multiclass.PNG)

       W doświadczeniach przejście do 2 etapu następowało przy oznaczeniu około 400-450 obrazów. Po przejściu do etapy *klastrowania*, aby jak najszybciej osiągnąć etap *wstępnego oznaczania* wystarczyło oznaczyć 100-200 zgrupowanych obrazów i poczekać, aż ruszy kolejny trening. Wtedy to liczba zdjęć *wstępnie oznaczonych* ulega zwiększeniu, a kiedy my dodatkowo potwierdzimy poprawność oznaczeń, wpłynie to pozytywnie na wyniki kolejnych treningów.

       Wyniki można wyeksportować na dwa sposoby:

       - dataset

         Takie rozwiązanie może w łatwy sposób umożliwić użycie naszego zbioru w kolejnych serwisach.

         ![](AML - screens\8_multiclass_dataset.PNG)

       - coco

         Plik wynikowy jest w formacie wynikowym, gdzie na początku pliku mamy opisane nasz zbiór danych:

         ![](AML - screens\9_multiclass_coco1.PNG)

         Następnie zapisane są oznaczenia:

         ![](AML - screens\9_multiclass_coco2.PNG)

       - ***OCENA DZIAŁANIA - 5/5*** 

         AML Data Labeling w tym rodzaju projektu bardzo dobrze radzi sobie z przyporządkowywaniem obrazów do poszczególnych klas. Oznaczenie dużych zbiorów obrazów przebiega bardzo szybko i przyjemnie. Fascynującym aspektem jest to, że kiedy model wstępnie oznaczy nam obrazy a my go będziemy utwierdzać w jego predykcjach, nasza praca ograniczy się wyłącznie do klikania przycisku *PRZEŚLIJ*. 

     - **Image Classification Multi-label**

       Projekt ten umożliwia oznaczenie obrazów wieloma klasami z zestawu klas. Obejmuje wszystkie omówione wcześniej 3 etapy.

       Przejście do drugiego etapu wymaga oznaczenie standardowo około 400-450 zdjęć. Pierwsze treningi i propozycje grupowania obrazów zdają się działać na tyle źle, że ciężko odróżnić dobór zdjęć w etapie grupowania od zdjęć w etapie ręcznego oznaczania. Po oznaczeniu w sumie ok 900-1000 zdjęć, zaobserwowano faktyczne grupowanie obrazów. Początkowo na 6/9 obrazach można było wyszukać wspólną klasę jednak podpowiedzi modelu były bardzo nieintuicyjne. Po osiągnięciu 1200 oznaczonych obrazów, klastrowanie przebiegało falowo. Kilka siatek 9 obrazów z bardzo dobrze wyselekcjonowanymi zdjęciami, po czym wiele siatek ze średnio pogrupowanymi obrazami.

       ![](AML - screens\17_work_multilabel.PNG)

       Podczas testów nie udało się osiągnąć etapu *wstępnego oznaczania*. Może mieć to wiele przyczyn, zaczynając od niejasnego zbioru danych, kończąc na błędach w oznaczaniu lub niedokładnym oznaczaniem obrazów. Brak osiągnięcia tego etapu może być również skutkiem tego, że trenowane modele nie do końca dobrze radzą sobie z problemem oznaczanie obrazów wieloma klasami.  Dla jednej klasy (poprzedni rodzaj projektu) wychodzi to świetnie.

       Używano do 5 klas, co można uznać za średnią ilość.  Naturalnie najlepiej grupowane były te zdjęcia, których oznaczeń wcześniej wystąpiło najwięcej.

       Jak w poprzednim projekcie dane możemy otrzymać na 2 sposoby:

       - dataset

         ![](AML - screens\10_multilabel_dataset.PNG)

       - coco

         ![](AML - screens\11_multilabel_coco1.PNG)

         ![](AML - screens\11_multilabel_coco2.PNG)

         - ***OCENA DZIAŁANIA - 4/5*** 

           AML dla tego projektu jest na pewno ułatwieniem w procesie oznaczania zdjęć. Ogromny wpływ na otrzymane wyniki ma zbiór danych oraz jego wstępne oznaczanie. Warto używać go dla projektów, gdzie trzeba oznaczyć tysiące zdjęć. W przypadku o ilości mniejszej niż 1000, nie ma sensu używać tego rodzaju projektu.

     - **Object Identyfication (Bounding Box)**

       Umożliwia przypisywania klasy oraz zdefiniowania obwiedni, czyli określenia dokładnego położenia reprezentanta danej klasy.

       Po początkowym ręcznym oznaczeniu 400-500 zdjęć, następuje przejście do etapu *wstępnego oznaczania*. W tym podejściu nie ma etapu klastrowania. ML oznacza elementy i radzi sobie z tym bardzo dobrze. 

       ![](AML - screens\18_work_obj.PNG)

       Naturalnie im lepiej zostaną oznaczone początkowe obrazy, tym w lepszy sposób tagowane są kolejne oraz tym szybciej można osiągnąć etap *wstępnego oznaczania*.

       Przeprowadzono test w którym nie oznaczano wszystkich elementów oraz nie robione tego w sposób dokładny. Sieć jednak była w stanie nauczyć się na podstawie tych wykonanych czynności i sama oznaczać zdjęcia w znacznie lepszy sposób, niż oznaczone ręcznie obrazy.

       Standardowo dane możemy otrzymać na 2 sposoby:

       - dataset

         ![](AML - screens\12_obj_ident.PNG)

       - coco

         Na początku są definiowane obrazy. W dalszej części zapisane są oznaczenia. Zapisana jest powierzchnia względem całego obrazu w skali 0-1. W *"bbox"* zapisana jest lokalizacja obiektu w postaci znormalizowanej:

         ```
         "bbox": [
         	topX,
         	topY,
         	szerokość obiektu,
         	wysokość obiektu
         ]
         ```

         ![](AML - screens\13_obj_ident_coco1.PNG)

         ![](AML - screens\13_obj_ident_coco2.PNG)

         - ***OCENA DZIAŁANIA - 5/5***

           Narzędzie to w znaczny sposób poprawia pracę nad oznaczaniem zbioru. Jeżeli mamy odpowiednie fundusze oraz mały zespół oraz zbiór liczący tysiące elementów, możemy poświęcić część zbioru. Możemy oznaczyć początkowe obrazy sposobem mniej dokładnym oraz oznaczać tylko elementy wyraźne i znaczące. Modelowi wystarczy to do treningu, po czym sam będzie proponował dokładniejsze oznaczenia. 

       - **Instance Segmentation (Polygon) (Preview)**

         Jest to format który nie wspiera Azure Machine Learning. Umożliwia oznaczanie obrazów w o wile bardziej dokładny sposób. Jednak brak ML, który by wspierał proces sprawia, że jest to tylko porządny edytor do oznaczania zdjęć.

         ![](AML - screens\19_work_poly.PNG)

         Standardowe uzyskanie wyników:

         - dataset

           ![](AML - screens\14_poly_dataset.PNG)

         - coco

           Oprócz oznaczenia ogólnego po przez *bbox*, mamy oznaczoną segmentację. Zapisane są tam wszystkie wierzchołki, które tworzą dany obszar po przez pary wartości X oraz Y.

           ![](AML - screens\15_poly_coco1.PNG)

           ![](AML - screens\15_poly_coco2.PNG)

           - ***OCENA DZIAŁANIA - 3.5/5***

             Jest to dobry edytor, który co najważniejsze umożliwia pracę w zespole nad jednym zbiorze danych. Ze względu na brak wsparcia AML, nie można ocenić go lepiej. Samo korzystanie z narzędzia jest bardzo intuicyjne. 

   - ***<u>WNIOSKI</u>***

     - Niezależnie od rodzaju projektu, sposób oznaczania obrazów jest bardzo intuicyjny i nie wymaga długiego szkolenia, żeby opanować pracę w tym środowisku.
     - W początkowych etapach projektu, pośpiech nie jest wskazany. Należy umożliwiać inicjacje kolejnych treningów i oczekiwać ich wyników, ponieważ dzięki takiemu działaniu jesteśmy w stanie szybciej osiągać satysfakcjonujące nas efekty.
     - Sens użycia narzędzia jest wtedy, kiedy pracujemy na względnie dużych zbiorach, liczących wiele tysięcy zdjęć. Praca na zbiorach poniżej 500 obraz kompletnie nie ma sensu.
     - Ze względu na to, że podczas treningu Maszyna Wirtualna pracuje tyle ile potrzebuje, trudno jest określić koszty jakie zostaną poniesione. W przeprowadzanych doświadczeniach zauważono, że znaczne wzrosty długości trwania poszczególnych treningów są dłuższe im wyższej iteracji jest to trening, ale nie dzieje się tak zawsze. Znając cenę pracy za godzinę, możemy tylko szacować. Im większy zbiór, tym należy spodziewać się dłuższej pracy maszyny.
     - Im bardziej zrównoważony zbiór (porównywalne ilości przedstawicieli z każdej klasy), tym cały proces przebiega bardziej płynnie i szybko
     -  Po stworzeniu projektu, jesteśmy w stanie dodatkowo rozdzielić proces **trenowania** oraz **prognozowania** na dwie oddzielne maszyny wirtualna - domyślnie pracują na tej samej, wyznaczonej przy tworzeniu projektu.

   - ZALETY

     - przyspieszenie procesu oznaczania zbioru obrazów
     - możliwość pracy w grupie na jednym zbierze
     - zarządzanie projektem i śledzenie postępów
     - dynamiczne powiększanie zbioru wejściowego
     - wsparcie Azure Machine Learning
     - Multi-class Image Classification oraz Object Identyfication

   - WADY

     - trudno w pełni określić koszty, można tylko szacować
     - Multi-label Image Classificatication oraz Instance Segmentation
     - nie ma możliwości powrotu do raz pominiętego obrazu 

2. ##### asddd

