# NeuronZadanie
Zadanie Rekrutacyjne Do KN Neuron

Rozwiązanie jest podzielone na dwie części

Zadanie rekrutacyjne do Koła Naukowego Neuron.

Projekt został podzielony na dwie części:

1. EDA

 - analiza sygnałów EEG oraz ich związku z klasyfikacją ADHD,

 - wyznaczenie nowych wskaźników na podstawie dostępnych danych,

 - zastosowanie metod statystycznych w celu wyodrębnienia najbardziej istotnych cech.

2. ModelEvaluation

 - porównanie różnych modeli i architektur klasyfikacyjnych,

 - dobór najlepszego rozwiązania na podstawie strojenia hiperparametrów,

 - ocena jakości modeli i wybór modelu o najlepszym dopasowaniu.

 ## Jak uruchomić projekt

Sklonuj repo

 ```{bash}

git clone https://github.com/NoMoreCookies/NeuronZadanie.git
cd NeuronZadanieRekrutacyjne

 ```

 Aktywuj wirtualne środowisko

 ```{bash}

python -m venv .venv
source .venv/bin/activate

 ```

Doinstaluj wymagane biblioteki

 ```{bash}

pip install -r requirements.txt

 ```

 W tym momencie powinieneś już móc odpalić EDA i ModelEvaluation np. z poziomu vscode.