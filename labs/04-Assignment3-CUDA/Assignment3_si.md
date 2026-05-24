Tukaj je prevod naloge v slovenščino:

---

# Molekularna dinamika: 2D simulacija Lennard-Jones

**Avtorji:** Uroš Lotrič, Davor Sluga
**Datum:** april 2026

---

## Uvod

Simulacija molekularne dinamike je splošno uporabljena računalniška tehnika za proučevanje fizikalnega obnašanja sistemov z veliko delci. Z numeričnim integriranjem enačb gibanja za vsak delec pod vplivom sil nam molekularna dinamika omogoča opazovanje pojavov, kot je samoorganizacija, ki nastanejo iz preprostih parnih interakcij.

V tej nalogi obravnavamo 2D sistem $N$ delcev, ki medsebojno delujejo prek Lennard-Jonesovega potenciala – klasičnega modela za žlahtne pline in druge preproste tekočine. Lennard-Jonesov potencial zajema tako kratkodometo odbojnost (zaradi prekrivanja elektronskih oblakov) kot tudi dolgodomet privlak (van der Waalsove sile) med parom delcev na razdalji $r$:

$$V(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

kjer je $\varepsilon$ globina potencialnega jaška in $\sigma$ končna razdalja, povezana z velikostjo atoma, pri kateri je potencial enak nič. V celotni simulaciji uporabljamo reducirane enote $m = \varepsilon = \sigma = 1$, kar poenostavi enačbe in izboljša numerično stabilnost.

Sila je enaka gradientu potenciala med parom delcev, $\mathbf{F} = -\nabla V(r)$. Če sta delca $i$ in $j$ na položajih $r_i$ in $r_j$, je njun vektor razmika $r_{ij} = r_i - r_j$. Z velikostjo vektorja $r_{ij} = |r_{ij}|$ in njenima projekcijama na kartezični koordinatni sistem $x_{ij}$ in $y_{ij}$ je sila delca $j$ na delec $i$ enaka:

$$\mathbf{F}_{ij} = \left(F_{xij}, F_{yij}\right) = F(r_{ij})\left( \frac{x_{ij}}{r_{ij}}, \frac{y_{ij}}{r_{ij}} \right)$$

kjer je

$$F(r) = 24\frac{\varepsilon}{r}\left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

velikost sile. Gibanje vsakega delca je pogojeno s silami vseh ostalih delcev. Skupna sila na delec $i$ je torej:

$$\mathbf{F}_{i} = \sum_{j \neq i} \mathbf{F}_{ij}$$

kar vodi do izračuna parnih sil v vsakem simulacijskem koraku s časovno zahtevnostjo $\mathcal{O}(N^{2})$.

---

## Simulacija

Referenčna implementacija v jeziku C skupaj z gradbenimi in zaganjajočimi skriptami je na voljo v [repozitoriju](https://github.com/laspp/HPC/blob/main/labs/04-Assignment3-CUDA/src/lennard-jones). Glavne komponente so opisane spodaj.

### Inicializacija delcev

Delci so postavljeni v obliki pravilne 2D mreže z opcijskim naključnim zamikom v sredini simulacijske škatle s stranico:

$$L = \frac{4}{3}\sqrt{\frac{N}{\rho}}$$

kjer je $\rho$ ciljna reducirana številska gostota. Hitrosti so naključno generirane, popravljene za odstranitev zanosa masnega središča in rescalirane, da ustrezajo enačbi:

$$E_k = \tfrac{1}{2}m\sum_i |\mathbf{v}_i|^2 = NT$$

kjer $T$ predstavlja ciljno reducirano temperaturo sistema.

### Izračun sil

Na vsakem simulacijskem koraku se vse parne sile znova izračunajo. Da bi se izognili interakcijam z neskončnim dosegom, se uporabi mejni radij $r_\text{cut} = 2{,}5\,\sigma$: pari z $r \geq r_\text{cut}$ ne prispevajo ničesar. Da bi odpravili nezveznost potenciala pri mejnem radiju in ohranili skupno energijo, se Lennard-Jonesov potencial nadomesti z zamaknjeno različico:

$$V_\text{zamaknjen}(r) = V(r) - V(r_\text{cut})$$

### Periodični robni pogoji

Simulacijska domena je kvadratna škatla s periodičnimi mejami – delci, ki zapustijo škatlo, vstopijo nazaj z nasprotne strani. Ker je simulacijska domena končna, a sistem posnema neskončnost, ima vsak delec neskončno mnogo periodičnih slik. Pri izračunu vektorja razmika $\mathbf{r}_{ij}$ vedno izberemo tisto sliko delca $j$, ki je najbližja delcu $i$. V praksi to dosežemo z zavitjem vsake komponente vektorja razmika v interval $(-L/2, L/2]$:

$$\mathbf{r}_{ij} = (\mathbf{r}_i - \mathbf{r}_j) - L \cdot \text{round}\left(\frac{\mathbf{r}_i - \mathbf{r}_j}{L}\right)$$

### Časovna integracija

Simulacija napreduje z metodo Leapfrog, ki je časovno obrnljiva in dobro ohranja energijo pri dolgih pogonjih. En simulacijski korak dolžine $\Delta t$ sestavljajo naslednje enačbe:

$$\mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right) = \mathbf{v}_i(t) + \tfrac{1}{2} \mathbf{a}_i(t) \Delta t$$
$$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right)\Delta t$$
$$\mathbf{a}_i(t+\Delta t) = \mathbf{F}_i(t+\Delta t) / m$$
$$\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right) + \tfrac{1}{2} \mathbf{a}_i(t + \Delta t)\Delta t$$

Na vsakem simulacijskem koraku se izračunata kinetična in potencialna energija sistema za preverjanje ohranitve skupne energije $E = E_k + E_p$.

---

## Naloga

Implementirajte vzporedno simulacijo Lennard-Jones v C/C++ z uporabo CUDA na podlagi referenčne implementacije. Algoritem mora delovati za poljubno število delcev in korakov. Opomba: koda v C je slabo optimizirana; jo prosto izboljšajte. Vsebuje tudi opcijsko kodo za generiranje animacij, ki jo lahko uporabite za pregled rezultatov.

### Organizacija referenčne kode

- `Makefile` – pravila za gradnjo projekta
- `run-lj.sh` – Sbatch skripta za pridobitev virov na gruči Arnes, gradnjo in zagon simulatorja
- `src/`
  - `main.c` – glavna datoteka projekta
  - `lennard-jones.cu` – koda simulacije Lennard-Jones
  - `gifenc.c` – koda za generiranje GIF animacij

### Osnovne naloge (za ocene 6–8)

- Vzporedno implementirajte algoritem z uporabo CUDA, čim bolj učinkovito. Izogibajte se nepotrebnim prenosom pomnilnika med gostiteljem in napravo. Pri razdelitvi dela poiščite optimalno število niti in velikost blokov niti. Omogočite možnost sledenja energiji sistema na vsakem koraku.
- Na gruči Arnes izmerite čas izvajanja algoritma za različna števila delcev: 1000, 2000, 4000 in 8000. Algoritem ocenite na 5000 simulacijskih korakih. Pri merjenju časa morajo biti vključeni tudi prenosi podatkov na GPU in z njega.
- Izračunajte pohitritev $S = t_s / t_p$ za vsako število delcev; $t_s$ je čas izvajanja zaporednega algoritma na CPE, $t_p$ pa čas izvajanja vzporednega algoritma na GPU. Algoritem zaženite večkrat (vsaj 5-krat) in povprečite meritve. Opomba: osnovna koda pri visokem številu delcev deluje dolgo; v takem primeru zadostuje en sam pogon.
- Vizualizirajte končno stanje sistema (ne vključite v poročilo, shranite ločeno). Ustvarite lahko tudi animacijo, ki prikazuje razvoj sistema skozi čas. Časa za ustvarjanje animacije ne vključite v meritve.
- Napišite kratko poročilo (1–2 strani), v katerem povzamete svojo rešitev in predstavite meritve na gruči. Poudarek naj bo na predstavitvi in razlagi časovnih meritev ter pohitritev.
- Kodo in poročilo (ena oddaja na par) oddajte na učilnico prek ustreznega obrazca do določenega roka (**5. 5. 2026**) in zagovarjajte kodo in poročilo na vajah.

### Bonus naloge (za ocene 9–10)

- Vzporedno implementirajte (z OpenMP) priloženo zaporedno kodo. Svojo izboljšano kodo (z optimalnim številom jeder) uporabite kot izhodišče za primerjalne meritve pohitritev z GPU.
- Izboljšajte referenčno kodo: po Newtonovem tretjem zakonu je vsaka sila enako velika in nasprotno usmerjena, zato zadostuje izračun le polovice interakcij: $N(N-1)/2$.
- Za nadaljnje zmanjšanje števila izračunanih interakcij razmislite o vodenju evidence sosedstev delcev. Delci, ločeni za več kot $r_\text{cut}$, se namreč ne vplivajo drug na drugega.
- Razmislite o razdelitvi dela med niti GPU in poiščite rešitev, ki GPU izkorišča čim bolje. Ena nit na delec morda ni optimalna možnost.
- Eksperimentirajte z načinom shranjevanja delcev v pomnilniku in kjer je smiselno, izkoristite deljeni pomnilnik.
- Za izvajanje simulacije uporabite dva GPU; razdelite delo med njima enakomerno in izmenjajte podatke po potrebi.

---

## HPC izziv

Ustvarite visoko optimizirano implementacijo kode simulacije Lennard-Jones za **3D primer**, ki generira rezultate skladne z referenčno kodo. Pripravite rešitev v C/C++, ki podpira grafične pospeševalnike z CUDA. Spodbujamo kombiniranje CUDE s sistemi z deljenim pomnilnikom prek knjižnice OpenMP.

Predlogovna koda in zaganjalne skripte so na voljo v repozitoriju. Vaša naloga je implementirati funkcijo `run_simulation` v datoteki `lennard-jones.cu`. Organizatorji bodo upoštevali le rešitve, zgrajene in izvedene z uporabo skripte `run-lj.sh`. Vsaka rešitev bo testirana v izoliranem okolju, ki sestoji iz enega 12-jedrnega vozlišča z dvema GPU Nvidia V100. Rešitve oddajte prek spletne strani predmeta.

Pri oddaji **NE ODDAJTE** datoteke `src/main.c`. Med testiranjem bomo zagotovili `main.c`, ki bo klicala funkcijo `run_simulation`.

- Testirali bomo na več konfiguracijah (1000+ delcev, 1000+ simulacijskih korakov). Beleženje energije bo onemogočeno.
- Datoteko `Makefile` in dodatne datoteke lahko spremenite, dokler se projekt uspešno prevede.
- Skripto `run-lj.sh` lahko spremenite; omejeni ste na eno vozlišče V100.
- Funkcija `run_simulation` naj vrne začetno in končno stanje sistema kot je definirano v strukturi `SimulationResult`. Pravilnost rešitve bomo preverili z dovolj majhnim odstopanjem vrednosti energij od referenčne rešitve.

### Pravila igre

- Delate lahko v parih ali samostojno; vsak sum na plagiatorstvo vodi do diskvalifikacije. Avtorje rešitve vpišite v datoteko `authors.txt`.
- Izziv je odprt do **nedelje, 31. maja**.
- Organizatorji bodo ocenjevali oddane rešitve glede pravilnosti in zmogljivosti (čas izvajanja). Teste bodo izvedli na gruči Arnes v izoliranem okolju.
- Rešitve bodo razvrščene glede na doseženo zmogljivost.
- **Nagrade:**
  - 1. mesto: bonus 10/10 popolno odgovorjenih vprašanj na pisnem izpitu
  - 2. mesto: bonus 7/10 popolno odgovorjenih vprašanj na pisnem izpitu
  - 3. mesto: bonus 5/10 popolno odgovorjenih vprašanj na pisnem izpitu
  - Organizatorji si pridržujejo pravico, da po lastni presoji nagradijo tudi druge rešitve.
- **Opombe:**
  - Kandidat lahko nagrade uveljavi le na prvem pisnem izpitu, ki ga opravi.
  - Končna ocena je povprečje ocene pisnega izpita in ocene nalog; ocena pisnega izpita določa zaokrožitev na celo število.