# Scan des actions — plan de trading swing

Chaque matin du lundi au vendredi, vers 4h15 (heure de Paris), GitHub Actions analyse les 676 actions de `tickers.csv` (S&P 500 et STOXX Europe 600) selon le « Plan de trading swing — règles universelles », à partir des données de Yahoo Finance (bibliothèque yfinance). Les résultats sont publiés dans `results/`, où la tâche Claude « Scan du matin » les lit à 6h45.

Éducatif, pas un conseil en investissement.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `scan_actions.py` | Le scan : filtres, score sur 10, setups A et B, stop, T1/T2, taille de position |
| `config.json` | Les règles chiffrées du plan (capital, risque, seuils). À modifier ici, pas dans le code |
| `tickers.csv` | La liste des actions, au format Yahoo (`SU.PA`, `AZN.L`, `BRK-B`…) |
| `results/scan_actions.json` | Résultat du jour, lu par la tâche Claude |
| `results/scan_actions.md` | Le même résultat, lisible dans GitHub |
| `results/fondamentaux.json` | Cache des données fondamentales (rafraîchies au plus tous les 7 jours) |
| `.github/workflows/scan.yml` | Le planning : du lundi au vendredi à 3h17 UTC |

## Ce que le scan applique

- **Filtres** : liquidité ≥ 2 M€/jour et capitalisation ≥ 1 Md€ ; ATR14 entre 1 % et 5 % ; pas de résultats dans les 7 jours ; free cash flow positif sur 2 des 3 derniers exercices et dette nette/EBITDA < 3 (hors services financiers) ; tendance mensuelle non baissière.
- **Setups** : A, cassure en clôture du dernier sommet de swing journalier non cassé ; B, rebond sur un creux de swing journalier récent (zone de 0,5 ATR) avec clôture au-dessus du plus haut de la veille, si la structure hebdomadaire est haussière.
- **Stop** : sous le niveau technique moins 0,5 ATR, entre 1 et 3 ATR. **T1** : première résistance majeure (sommet de swing hebdomadaire ou plus haut 52 semaines), à au moins 2R. **T2** : la suivante.
- **Score** : les 9 critères du plan. 7 et plus = entrée validée ; 5–6 = trade papier.
- **Taille** : 1 % de 30 000 € par trade, plafond de 25 % du capital par position, devises converties en euros.

**Approximations** : Yahoo ne fournit ni order blocks, ni FVG, ni BOS/CHoCH, ni notation TradingView. Les pivots et zones sont calculés avec les sommets et creux de swing ; la notation mensuelle avec le MACD mensuel et la SMA 10 mensuelle.

## Limites

- yfinance n'est pas une source officielle : Yahoo peut bloquer temporairement les requêtes (erreur 429). Le script attend et réessaie ; une action sans données est listée dans `couverture.sans_donnees`.
- Le cache des fondamentaux se remplit progressivement : 150 fiches par jour, en commençant par les actions qui ont un setup. Les premiers jours, certaines vérifications apparaissent en « à vérifier ».
- GitHub peut lancer un workflow planifié avec du retard. Si le résultat n'est pas daté du jour, la tâche Claude le signale.

## Lancer le scan à la main

Onglet **Actions** → **Scan des actions** → **Run workflow**.
