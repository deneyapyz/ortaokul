import numpy as mat
import skfuzzy as bulanik
from skfuzzy import control as kontrol

yaya_sayisi = kontrol.Antecedent(mat.arange(1, 31, 1), 'yaya sayısı')
bekleme_suresi = kontrol.Consequent(mat.arange(1, 61, 1), 'geçiş zamanı')

yaya_sayisi.automf(3)
bekleme_suresi.automf(3)

kural1 = kontrol.Rule(yaya_sayisi['poor'] , bekleme_suresi['poor'])
kural2 = kontrol.Rule(yaya_sayisi['average'], bekleme_suresi['average'])
kural3 = kontrol.Rule(yaya_sayisi['good'] , bekleme_suresi['good'])

kontrol_sure = kontrol.ControlSystem([kural1, kural2, kural3])
sure = kontrol.ControlSystemSimulation(kontrol_sure)

sure.input['yaya sayısı'] = 10
sure.compute()
print (sure.output['geçiş zamanı'])
bekleme_suresi.view(sim=sure)
