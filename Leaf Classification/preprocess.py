# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/8 上午10:27

import os
import pandas as pd
classes = ['Acer_Capillipes', 'Acer_Circinatum', 'Acer_Mono', 'Acer_Opalus', 'Acer_Palmatum', 'Acer_Pictum', 'Acer_Platanoids', 'Acer_Rubrum', 'Acer_Rufinerve', 'Acer_Saccharinum', 'Alnus_Cordata', 'Alnus_Maximowiczii', 'Alnus_Rubra', 'Alnus_Sieboldiana', 'Alnus_Viridis', 'Arundinaria_Simonii', 'Betula_Austrosinensis', 'Betula_Pendula', 'Callicarpa_Bodinieri', 'Castanea_Sativa', 'Celtis_Koraiensis', 'Cercis_Siliquastrum', 'Cornus_Chinensis', 'Cornus_Controversa', 'Cornus_Macrophylla', 'Cotinus_Coggygria', 'Crataegus_Monogyna', 'Cytisus_Battandieri', 'Eucalyptus_Glaucescens', 'Eucalyptus_Neglecta', 'Eucalyptus_Urnigera', 'Fagus_Sylvatica', 'Ginkgo_Biloba', 'Ilex_Aquifolium', 'Ilex_Cornuta', 'Liquidambar_Styraciflua', 'Liriodendron_Tulipifera', 'Lithocarpus_Cleistocarpus', 'Lithocarpus_Edulis', 'Magnolia_Heptapeta', 'Magnolia_Salicifolia', 'Morus_Nigra', 'Olea_Europaea', 'Phildelphus', 'Populus_Adenopoda', 'Populus_Grandidentata', 'Populus_Nigra', 'Prunus_Avium', 'Prunus_X_Shmittii', 'Pterocarya_Stenoptera', 'Quercus_Afares', 'Quercus_Agrifolia', 'Quercus_Alnifolia', 'Quercus_Brantii', 'Quercus_Canariensis', 'Quercus_Castaneifolia', 'Quercus_Cerris', 'Quercus_Chrysolepis', 'Quercus_Coccifera', 'Quercus_Coccinea', 'Quercus_Crassifolia', 'Quercus_Crassipes', 'Quercus_Dolicholepis', 'Quercus_Ellipsoidalis', 'Quercus_Greggii', 'Quercus_Hartwissiana', 'Quercus_Ilex', 'Quercus_Imbricaria', 'Quercus_Infectoria_sub', 'Quercus_Kewensis', 'Quercus_Nigra', 'Quercus_Palustris', 'Quercus_Phellos', 'Quercus_Phillyraeoides', 'Quercus_Pontica', 'Quercus_Pubescens', 'Quercus_Pyrenaica', 'Quercus_Rhysophylla', 'Quercus_Rubra', 'Quercus_Semecarpifolia', 'Quercus_Shumardii', 'Quercus_Suber', 'Quercus_Texana', 'Quercus_Trojana', 'Quercus_Variabilis', 'Quercus_Vulcanica', 'Quercus_x_Hispanica', 'Quercus_x_Turneri', 'Rhododendron_x_Russellianum', 'Salix_Fragilis', 'Salix_Intergra', 'Sorbus_Aria', 'Tilia_Oliveri', 'Tilia_Platyphyllos', 'Tilia_Tomentosa', 'Ulmus_Bergmanniana', 'Viburnum_Tinus', 'Viburnum_x_Rhytidophylloides', 'Zelkova_Serrata']

train_txt = open('train.txt','w')
train_csv = pd.read_csv(r'leaf-classification/train.csv')
ids = train_csv['id']
species = train_csv['species']

for i in range(len(ids)):
    train_txt.write(str(ids[i]))
    train_txt.write(' ')
    train_txt.write(str(classes.index(str(species[i]))))
    train_txt.write('\n')
train_txt.close()

test_txt = open('test.txt','w')
test_csv = pd.read_csv(r'leaf-classification/test.csv')
ids = test_csv['id']
for i in range(len(ids)):
    test_txt.write(str(ids[i]))
    test_txt.write('\n')
test_txt.close()

