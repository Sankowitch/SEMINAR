from json.encoder import INFINITY
import numpy as np
from numpy import linalg as LA
from gurobipy import *
import gurobipy as gp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
from IPython.display import HTML, Image
feasible = False


#svugdje gdje su ograničenja označeno sa "OGRANIČENJA __"

def calculateK(T, h):
    #k mora biti iz Z
    delta = 0.01
    k = 0.1
    while (k.is_integer() == False):
        k = T/h + 1
        k = round(k, 5)
        h = h + delta
    return (int(k), h-delta)

def generateMatrix(a, b):
    matrix = np.zeros((a, b))
    return matrix


#12N x 3NK 
#initial akc
#final pose
#velocity
#akc
#prvo za letjelicu 1 sve, pa za letjelicu 2 sve
#provjerena matrica-OK!
def generateAeq(N, k, h):
    a = 12* N
    b = 3 * N * k
    #generiramo nul matricu koju popunjavamo
    A_eq = generateMatrix(a, b)
    
    counter = 0  #potreban za initial akc
    pocetak = 0  #potreban za final pose da znamo gdje je prva akceleracija od vozila i u nekom smjeru
    pocetak_dva = 0 #isto kao ovo gore samo za velocity
    pocetak_tri = 3*N*(k-1) #isto kao gore ali za akc

    for i in range (0, a):
        uvjet = i % 12
        if (uvjet <= 2):
            #punimo initial akc
            A_eq[i, counter] = 1
            counter = counter + 1
        if (uvjet >= 3 and uvjet <= 5):
            #punimo final pose
            neparni_broj = 3
            razlika = 2*k - neparni_broj
            delta = 0
            while (razlika >= 1):
                A_eq[i, pocetak+delta] = ((h * h)/2) * razlika
                delta = delta + 3 * (N)      #3 za svaki smjer od sljedecih vozila zato je N-1 i dva kao ostali smjerovi od itog 
                neparni_broj = neparni_broj + 2
                razlika = 2*k - neparni_broj
            pocetak = pocetak + 1
        if (uvjet >= 6 and uvjet <= 8):
            #punimo velocity
            a = 0
            delta = 0
            while (a < k-1):
                a = a+ 1
                A_eq[i, pocetak_dva + delta] = h
                delta = delta + 3 * (N) 
            pocetak_dva=pocetak_dva + 1
        if (uvjet >= 9):
            #punimo akceleraciju, finalnu
            A_eq[i, pocetak_tri]=1
            pocetak_tri = pocetak_tri + 1
           
    return A_eq
        
#12N X 1
#initial akc
#final pose
#velocity
#akc
#prvo za letjelicu 1 sve, pa za letjelicu 2 sve
#provjerena matrica-OK!
def generateBeq(N , k, h, pocetnePozicije, finalnePozicije, pocetneBrzine, finalneBrzine, pocetneAkceleracije):
    a = 12* N
    b_eq = np.zeros(a)
    index = 0
    index_dva = 0
    index_tri = 0
    g = 9.81
    for i in range(0, a):
        uvjet = i % 12
        if (uvjet<=2):
            #initial akc
            b_eq[i] = pocetneAkceleracije[uvjet]
        if (uvjet >= 3 and uvjet <= 5):
            #punimo final pose
            b_eq[i] = finalnePozicije[index] - pocetnePozicije[index] - h * (k-1) * pocetneBrzine[index]
            index = index + 1
        if (uvjet >= 6 and uvjet <= 8):
            #final velocity
            b_eq[i] =  finalneBrzine[index_dva] -  pocetneBrzine[index_dva]
            index_dva = index_dva + 1
        if (uvjet >= 9):
            akc = 0
            if (index_tri == 2):
                index_tri = 0
                akc = g
                #finalna akceleracija je 9.81
                
            else:
                index_tri = index_tri + 1
            b_eq[i]= akc
            
        
    return b_eq

#M x 3NK
#position -> 6NK - 3 (za svaki msjer) *2 (donja i gornja granica) * N za svako vozilo * K za svaki korak K
#velocity -> ne ogranicavamo
#akc -> 6NK - 3 (za svaki msjer) *2 (donja i gornja granica) * N za svako vozilo * K za svaki korak K
#jerk
#dalje su ofranicenja za pozcije medjusobno
def generateAin(M, N, k, h):
    A_in = generateMatrix(M, 3*N*k )
    index_k = k
    pocetak = 0
    umnozak = -1
    pocetak_akc = 0
    umnozak_akc = -1
    index_jerk = 0
    umnozak_jerk = -1
    j = 0 #za jerk
    for i in range(0, M):
        neparniBrojMax = 2*k - 1
        if (i < 6*  N * k):
            #punimo pozicije prvo, prvo sve pozicije za smjer x pa z, y od K do 1 itd i to dva puta, jednom za pMIN ogranicenje a drugi put za pMax ogranicenje 
            #za pmin  vrijedi -p <= -pmin zato upisujemo sa minusima
            #od nazad jer mi je tako lakse znaci prvo za k = K
            if (i == 3 * N * k):
                #druga tura punjenja, punimo opet isto samo sa drugim predznakom
                umnozak = 1
                pocetak = 0
                index_k = k
            neparni_broj = 3
            razlika = 2*index_k - neparni_broj
            delta = 0
            if (index_k != 1):
                while (razlika >= 1):
                    A_in[i, pocetak+delta] = ((h * h)/2) * razlika * umnozak
                    delta = delta + 3 * (N)      #3 za svaki smjer od sljedecih vozila zato je N-1 i dva kao ostali smjerovi od itog 
                    neparni_broj = neparni_broj + 2
                    razlika = 2*index_k - neparni_broj
            else:
                #kad smo dosli do k = 1, dosli smo do zadnjeg za taj smjer -> pocetak treba promjeniti 
                pocetak = pocetak + 1
                index_k = k
                continue
            index_k = index_k - 1
        if ( i >= 6 * N * k and i < 12 * N * k):
            #akceleracija
            if (i == 9 * N * k):
                #na pola smo, ispocetka punimo
                pocetak_akc = 0
                umnozak_akc = 1
            A_in[i, pocetak_akc] = 1 * umnozak_akc
            pocetak_akc = pocetak_akc + 1
        #gornja granica 12Nk + N * k * 3 (smjer) * 2(za min i za max, 2 put punimo)
        if (i >= 12 * N * k and i < (18 * N * k)):
            #punimo jerk
            if (index_jerk >= 3 * k ):
                #popunili smo za neki smjer
                #popunili smo sve za smjer x ili y ili z 
                j = j + 1
                if (j == 3 * N):
                    j = 0
                #dosli smo do kraja prvog punejna x y z
                index_jerk = 0
            if (i == 15 * N * k):
                #na pola smo
                #punimo ispocetka
                index_jerk = 0
                j = 0
                umnozak_jerk = 1
            if (index_jerk == 0):
                index_jerk = index_jerk + 3 * N
                continue
            
            A_in[i, index_jerk + j] = 1/h * umnozak_jerk
            A_in[i, index_jerk + j - 3*N] = -1/h * umnozak_jerk
            
            index_jerk = index_jerk + 3 * N
            
    return A_in

def generateBin(M, N, k, h, R,  pocetnePozicije, krajnePozicije, pocetneBrzine, ogradaAkcDonja, ogradaAkcGornja, ogradaDonjaJerk, ogradaGornjaJerk, pozicijeMin, pozicijeMax):
    b_in = np.zeros(M)
    index_k = k
    index_k_dva = k
    index_pose = 0
    index_pose_dva = 0
    jerk_counter =0
    jerk_counter_dva = 0
    index_p_min = 0
    index_p_max = 0
    j = 0
    l = 0
    m = 0 
    m_dva = 0
    prvi = True
    prvi_dva = True
    for i in range (0, M):
        if (i < 3 * N * k):
            #ubacujemo p min, prvo k puta ubacujemo p min x za prvo vozilo, pa p min y za prvo itd
            if (i % k == 0 and i != 0):
                index_pose = index_pose + 1
                index_k = k
            #za k = 1 ovo dole ce se i onako mnoziti s nulom
            #minimalna poza je 0
            # -p <= -p_min
            # -p - pi[1] - h(k-1)*vi[1] <=  - pmin
            # -p <= -pmin + pi[1] + h(k-1)*vi[1] 
            if (index_k == 1):
                continue
            #p min i max su upper i lower bounds 
            if (index_p_min  % 3 == 0):
                index_p_min = 0
            p_min = pozicijeMin[index_p_min]
            index_p_min = index_p_min + 1
            b_in[i] =   -p_min +  h*(index_k - 1) * pocetneBrzine[index_pose] + pocetnePozicije[index_pose]
           
            index_k = index_k - 1

        if (i >= 3 * N * k and i < 6 * N * k):
            #ubacujemo p max
            if (i % k == 0 and i!= 3* N * k):
                index_pose_dva = index_pose_dva + 1
                index_k_dva = k
            
            if (index_p_max % 3 == 0):
                index_p_max = 0
            
            p_max = pozicijeMax[index_p_max]
            index_p_max = index_p_max + 1
            b_in[i] = p_max  - h*(index_k_dva - 1) * pocetneBrzine[index_pose_dva] - pocetnePozicije[index_pose_dva]
            index_k_dva = index_k_dva - 1
        
        if (i >= 6*N*k and i < 9*N*k):
            #punimo akc donja granica
            if (j == 3):
                j= 0
            # -a <= -amin
            b_in[i] = - ogradaAkcDonja[j] 
            j = j + 1
        if (i >= 9*N*k and i < 12*N*k):
            #punimo akc gornja granica
            if (l == 3):
                l= 0
            # a <= amax
            b_in[i] =  ogradaAkcGornja[l] 
            l = l + 1
        if (i >= 12 * N * k and i < 15 * N * k ):
            #punimo donje granice za jerk
            #pune se kao vozilo 1 sve x-evi, vozilo 1 sve y, vozilo 1 sve z, oa vozilo 2, x 
            if (jerk_counter == 0 and prvi == True):
                jerk_counter = jerk_counter + 1
                prvi = False
                continue
            if (jerk_counter == k):
                jerk_counter =  0 + 1
                m = m + 1
                if (m == 3):
                    m = 0
                continue
            b_in[i] = -1 * ogradaDonjaJerk[m]
            jerk_counter = jerk_counter + 1

        if (i >= 15 * N * k and i < (18 * N * k)):
            #punimo gornje granice za jerk
            if (jerk_counter_dva == 0 and prvi_dva == True):
                jerk_counter_dva = jerk_counter_dva + 1
                prvi_dva = False
                continue
            if (jerk_counter_dva == k):
                jerk_counter_dva =  0 + 1
                m_dva = m_dva + 1
                if (m_dva == 3):
                    m_dva = 0
                continue
            b_in[i] =  ogradaGornjaJerk[m_dva]
            jerk_counter_dva = jerk_counter_dva + 1 
            
            
        if (i >= (18 * N * k)):
           b_in[i] =  R
           #b_in[i] = 0
          
            
    return b_in

# n(n+1)/2 samo sam ja dumb
def zbroj(N):
    broj = 0
    for i in range(1, N):
        broj = broj + (N - i) 
    return broj

#u A_in upisujemo dodatna ogranicenja (ogranicenja 17)
#x je psi
#R je udaljenost minimalna letjelica medjusobno
def nadopuni(A_in, b_in, h,  N, k, x, R, M, pocetnePoz, pocetneBrzine):
    #punimo od 18 *  N * k retka do +3NK , znaci 3 za svaki smjer, k za svaki korak i svako vozilo sa svakim -> N - 1 * N - 2 ...
    pocetak = 18 * N * k
    kraj = pocetak + zbroj(N) * k  #zbroj je Nc u članku
    
    if (N == 1):
        #ako je broj letjellica 1 nemamo sto racunati
        return (0, 0)
    index1 = 0  #index pocetne pozicije vozila 1 (dakle x kod p1(x, y, z))
    index2 = 3  #index pocetne pozicije vozila 2 (dakle x kod p1(x, y, z))

    indexPodvektorPOCETNI1 = 0  #index akceleracija
    indexPodvektorPOCETNI2 = 3  #index akceleracija
    counter = 0


    
    k_pocetni = 1
    for i in range (pocetak, kraj):
        #pq1 i 2 ELEMENT R3
        pq1 = np.zeros(3)
        pq2 = np.zeros(3)

        

        if (k_pocetni == k+1):
            #dosli smo do kraja popunjavanja 1 i 2, sada treba popunjavati 1 i 3 itd...
            k_pocetni = 1
            counter = counter + 1
            if (counter == N-1):
                #OBAVILI smo sve kombinacije 1 i 2, 1 i 3 ... 1 i n, sada treba prebaciti na 2 i 3
                #index1 pomicemo sa 1 na 2 ili sa 2 na 3 itd
                #index2 uvijek je sljedeci poslije indexa1 dakle index1 + 3
                index1 = index1 + 3
                index2 = index1 + 3
                indexPodvektorPOCETNI1 = indexPodvektorPOCETNI1 + 3
                indexPodvektorPOCETNI2 = indexPodvektorPOCETNI1 + 3
            else:
                #ako nismo popunili sve kombinacije 1 i nesto, tada taj nesto dignemo
                index2 = index2 + 3
                indexPodvektorPOCETNI2 = indexPodvektorPOCETNI2 + 3

        indexPodvektor1 = indexPodvektorPOCETNI1
        indexPodvektor2 = indexPodvektorPOCETNI2
   
        if (counter == N):
            break
        
        #popunjavanje pq1 i pq2
        #prvo popunjavamo p[1] + h(k-1)v[1]
        for j in range(0, 3):
            pq1[j] = pocetnePoz[index1 + j] + h*(k-1)* pocetneBrzine[index1 + j]
            pq2[j] = pocetnePoz[index2 + j] + h*(k-1)* pocetneBrzine[index2 + j]
        
        if (k_pocetni >= 2):
            #onda upisujemo ono sa akceleracijama
            podvektor1 = np.zeros(3)
            podvektor2 = np.zeros(3)
            neparni_broj = 3
            koeficijent = 2*k_pocetni - neparni_broj
            while(koeficijent >= 1):
                podvektor1 = x[indexPodvektor1:indexPodvektor1+3]
                podvektor2 = x[indexPodvektor2:indexPodvektor2+3]
                
                podvektor1 = (((h*h)/2) * koeficijent) * podvektor1
                podvektor2 = (((h*h)/2) * koeficijent) * podvektor2
                
                pq1 = np.add(pq1, podvektor1)
                pq2 = np.add(pq2, podvektor2)
                neparni_broj = neparni_broj + 2
                koeficijent = 2*k_pocetni - neparni_broj
                indexPodvektor1 = indexPodvektor1 + 3*N
                indexPodvektor2 = indexPodvektor2 + 3*N
        
        
        #pq1 - pq2
        razlika = np.subtract(pq1, pq2)
        #||pq1 - pq2||2
        normiran = LA.norm(razlika)
        if (normiran == 0):
            return (0, 0)
        #ni = pq1 - pq2/||pq1 - pq2||2
        ni = razlika / normiran
        
        
       
       
        #||pq1 - pq2||2 + ni((p1[k]-p2[k]) - (pq1[k]-pq2)[k]) >= R
        #ni[(p1[k]-p2[k]) - (pq1[k]-pq2[k])] >= R - ||pq1 - pq2||2
        #- ni[(p1[k]-p2[k]) - (pq1[k]-pq2[k])] <= -R + ||pq1 - pq2||2
        #- ni(p1[k]-p2[k]) + ni(pq1[k]-pq2[k]) <= -R + ||pq1 - pq2||2
        #- ni(p1[k]-p2[k]) <= -R + ||pq1 - pq2||2 - ni(pq1[k]-pq2[k])
        #p1 i p2 su pocetne pozicije tih dvaju vektora (p1x, p1y, p1z), a v1 i v2 pocetne brzine
        # -ni [(p1 + (k-1)hv1 + h2/2(akc1)) - (p2 + (k-1)hv2 + h2/2(akc2))] <= -R + ||pq1 - pq2||2 - ni(pq1[k]-pq2[k])
        # -ni[(p1 + (k-1)hv1 -p2 - (k-1)v2)  + (h2/2(akc1) - h2/2(akc2))] <= -R + ||pq1 - pq2||2 - ni(pq1[k]-pq2[k])
        # -ni(p1 + (k-1)hv1 -p2 - (k-1)v2) -ni(h2/2(akc1) - h2/2(akc2)) <= -R + ||pq1 - pq2||2 - ni(pq1[k]-pq2[k])
        #-ni(h2/2(akc1) - h2/2(akc2))  <= -R + ||pq1 - pq2||2 - ni(pq1[k]-pq2[k]) + ni(p1 + (k-1)hv1 -p2 - (k-1)v2)
        #desna strana ide u b_in matricu
        #vektor = (p1 + (k-1)hv1 -p2 - (k-1)v2
        #vektor = p1 - p2 + (k-1)h[v1 - v2]
        p1 = pocetnePoz[index1: index1+3]
        p2 = pocetnePoz[index2: index2+3] 
        v1 = pocetneBrzine[index1: index1+3]
        v2 = pocetneBrzine[index2: index2+3]
        vektor1 = np.subtract(p1, p2)
        vektor2 = np.subtract(v1, v2)
        vektor2 = (k_pocetni - 1)*h * vektor2
        
        
        b_in[i] = -R + normiran - ni.dot(razlika) +ni.dot(np.add(vektor1, vektor2))

        #lijeva strana ide u A_in matricu
        # -ni * (h*h/2) * [(2k-3)a1[1] - (2k-3)a2[1] + .... + a1[k-1] -a2[k-1] ]

        indexPodvektor1 = indexPodvektorPOCETNI1
        indexPodvektor2 = indexPodvektorPOCETNI2

        if (k_pocetni >= 2):
            #ako je k = 1 akc se ne upisuju
            #-ni * (h*h/2)
            vektor = - 1 * ((h*h)/2) * ni
            neparni_broj = 3
            koeficijent = 2*k_pocetni - neparni_broj
            while(koeficijent >= 1):
                for l in range(0 , 3):
                    #koeficijent * vektor * akc
                    # (2k - neparni_broj) * [(vek1, vek2, vek3) * (ax, ay, az)] * (+-1)
                    # za akc1 ide +1, za akc2 ide -1 (jer su uz a1 uvijek +, a a2 se oduzima, pogledaj komentare iznad)
                    #(2k - neparni_broj)  * [vek1*ax + vek2*ay  + vek3*az] * (+-1)
                    A_in[i, indexPodvektor1 + l ] = vektor[l] * koeficijent * 1
                    A_in[i, indexPodvektor2 + l ] = vektor[l] * koeficijent * (-1)
                indexPodvektor1 = indexPodvektor1 + 3*N
                indexPodvektor2 = indexPodvektor2 + 3*N
                neparni_broj = neparni_broj + 2
                koeficijent = 2*k_pocetni - neparni_broj
        
       
        k_pocetni = k_pocetni + 1    
        
    return (A_in, b_in)

#q-iteracija
def optimize(A_eq, b_eq, A_in, b_in, N, k, T, g, prviUlaz):
    q_model = gp.Model('quadratic')
    value = 0
    size = 3*N*k
    #ogranicenje 12 se moze dodati kao lb i ub 
    x = q_model.addMVar(size, lb = -1000000,  ub = 1000000, vtype=gp.GRB.CONTINUOUS)

    #minimize f(x) = xPx +qx + r
    # f(x) = EE||ai[k]+g||**2
    #||ai[k]+g||**2 = ||(aix[k], aiy[k] , aiz[k]) + (0 , 0, g)||**2
    # = ||(aix[k], aiy[k], aiz[k]+ g)||**2 = aix[k]**2 + aiy[k]**2 + (aiz[k] + g)**2
    #= aix[k]**2 +  aiy[k]**2 +  aiz[k]**2 + 2 aiz[k]g + g**2
    #xPx =  aix[k]**2 +  aiy[k]**2 +  aiz[k]**2 (i tako sa sve akceleracije svih vozila u svakom k)
    # => matrica P su samo jedinice
    P = np.ones((3*N*k, 3*N*k))
    #qx =  2 aiz[k]g (i tako sa svaki n i k, ovdje imamo samo z smjerove, dakle jedinice ce biti u svakom trecem stupcu)
    #matrica qx je 2*g*[0, 0, 1, 0, 0, 1, ..., 0, 0, 1]
    q = np.zeros(3*N*k)
    for i in range(0, 3*N*k):
        #i je od 0 pa zato +1
        if ((i+1)% 3 == 0):
            q[i] = 1
    q = 2*g*q
    #r = g**2 (ali sa svaki N i svaki k=> r = Nk*g**2)
    r = N*k*g*g
    #nije potrebno transponirati g. matricu to ovo samo hendla (vidi ptimjer sesti)
    q_model.setObjective(x  @ P @ x + q @ x + r, gp.GRB.MINIMIZE )
 
    #AeqX = b_eq   i A_inX <= b_in
    q_model.addConstr(A_eq @ x== b_eq)
    #q_model.addConstr(A_eq @ x == np.ones(12*N))
    #u prvom ulazu radimo samo ograničenja jednakosti, bez ograničenja nejednakosti
    if (prviUlaz == False):
        
        q_model.addConstr(A_in @ x <= b_in)

    if (prviUlaz == True):
        prviUlaz = False

    

    q_model.setParam('OutputFlag' , False)
    
    q_model.optimize()
    
    print("status")
    print(q_model.status)
    
    x_mat = np.zeros(size)
    if (q_model.status == 2):
        print("nađeno riješenje!")
        i = 0
        value =  q_model.objVal
        for v in q_model.getVars():
            x_mat[i] = v.x
            i = i + 1
        #print("-----X MATRICA------")
        #printMatrix(x_mat)
    else:
        print("nije nađeno riješenje...")
   
    return (x_mat, value, q_model.status)


#crtanje svake od trajekotrija za letjelice
def nacrtaj(x, N, k,h, pocetnePoz, pocetneBrzine ):

    index = 0
    indeX_akcPOCETNI = 0
    ax = plt.axes(projection = "3d")
    
    ax.set_xlabel("X vrijednosti")
    ax.set_ylabel("Y vrijednosti")
    ax.set_zlabel("Z vrijednosti")
    rijecnik = {}
    for i in range(0, N):
        #za svaku letjelicu skupljamo x y i z koordinate
        x_data = np.ones(k)
        y_data = np.ones(k)
        z_data = np.ones(k)
        listaTocki=[]

        ax.scatter(pocetnePoz[index], pocetnePoz[index+1], pocetnePoz[index+2], marker="s")

        k_pocetni = 1
        #p[k] = p[1] + h(k-1)*v[1] + (h**2/2)*[(2k-3)*a[1] + (2k-5)*a[2] + ... + a[k-1]]
        for j in range(0, k):
            x_data[j] = pocetnePoz[index] + h*(k_pocetni-1) * pocetneBrzine[index]
            y_data[j] = pocetnePoz[index+1] + h*(k_pocetni-1) * pocetneBrzine[index+1]
            z_data[j] = pocetnePoz[index+2] +h*(k_pocetni-1) * pocetneBrzine[index+2]

            index_akc = indeX_akcPOCETNI 
            if (k_pocetni >=2 ):
                #dodajemo ovo sa akceleracijama
                neparni_broj = 3
                koeficijent = 2*k_pocetni - neparni_broj
                while (koeficijent >= 1):
                    x_data[j] = x_data[j] + ((h*h)/2) * koeficijent * x[index_akc]
                    y_data[j] = y_data[j] + ((h*h)/2) * koeficijent * x[index_akc+1]
                    z_data[j] = z_data[j] + ((h*h)/2) * koeficijent * x[index_akc+2]
                    index_akc = index_akc + 3 * N
                    neparni_broj = neparni_broj + 2
                    koeficijent = 2*k_pocetni - neparni_broj
            k_pocetni = k_pocetni + 1    
       
        #ax.plot(x_data, y_data, z_data) 
        
        plt.pause(0.2)  
        listaTocki.append(x_data)
        listaTocki.append(y_data)
        listaTocki.append(z_data)
        rijecnik[i] = listaTocki
        
        
        #idemo na sljedecu trajektoriju
        index = index + 3
        indeX_akcPOCETNI = indeX_akcPOCETNI + 3
    #plt.show() 

    xevi = []
    yoni = []
    zeovi = []
    x_plot = {}
    y_plot = {}
    z_plot= {}

    for i in range(0, k):
        
        listaPomocX = []
        listaPomocY = []
        listaPomocZ = []
        
        for j in range(0, N):
            listaPomocX = []
            listaPomocY = []
            listaPomocZ = []
            lista = rijecnik[j] 
           
            x_data = lista[0]
            y_data = lista[1]
            z_data = lista[2]
            
            xevi.append(x_data[i])
            yoni.append(y_data[i])
            zeovi.append(z_data[i])
            if (i != 0):
                listaPomocX = x_plot[j]
                listaPomocY = y_plot[j]
                listaPomocZ = z_plot[j]
            listaPomocX.append(x_data[i])
            listaPomocY.append(y_data[i])
            listaPomocZ.append(z_data[i])
            x_plot[j] =  listaPomocX
            y_plot[j] =  listaPomocY
            z_plot[j] =  listaPomocZ

        

        #ax.scatter(xevi, yoni, zeovi) 
        
        if (i != 0):
            for l in range(0, N):
                x = []
                y = []
                z = []
                x_ = x_plot[l]
                y_ = y_plot[l]
                z_ = z_plot[l]
                for m in range(0, len(x_)):
                    x.append(x_[m])
                    y.append(y_[m])
                    z.append(z_[m])
                
                ax.plot(x,y,z)
                
                
            
        #plt.pause(0.2) 
    
    
    plt.show()  
    

 
def dvaDcrtanje(x, N, k,h, pocetnePoz, pocetneBrzine, finalnePozicije):
    index = 0
    indeX_akcPOCETNI = 0
    #rc('animation', html='html5')
    ax = plt.axes()
    plt.xlim(-70, 70)
    plt.ylim(-70,70)
    ax.set_xlabel("X vrijednosti")
    ax.set_ylabel("Y vrijednosti")
    #ax.set_zlabel("Z vrijednosti")
    rijecnik = {}
    for i in range(0, N):
        #za svaku letjelicu skupljamo x y i z koordinate
        x_data = np.ones(k)
        y_data = np.ones(k)
     
     #   z_data = np.ones(k)
        listaTocki=[]

        pocetna = ax.scatter(pocetnePoz[index], pocetnePoz[index+1], marker="s")
        

        k_pocetni = 1
        #p[k] = p[1] + h(k-1)*v[1] + (h**2/2)*[(2k-3)*a[1] + (2k-5)*a[2] + ... + a[k-1]]
        for j in range(0, k):
            x_data[j] = pocetnePoz[index] + h*(k_pocetni-1) * pocetneBrzine[index]
            y_data[j] = pocetnePoz[index+1] + h*(k_pocetni-1) * pocetneBrzine[index+1]
            #z_data[j] = pocetnePoz[index+2] +h*(k_pocetni-1) * pocetneBrzine[index+2]

            index_akc = indeX_akcPOCETNI 
            if (k_pocetni >=2 ):
                #dodajemo ovo sa akceleracijama
                neparni_broj = 3
                koeficijent = 2*k_pocetni - neparni_broj
                while (koeficijent >= 1):
                    x_data[j] = x_data[j] + ((h*h)/2) * koeficijent * x[index_akc]
                    y_data[j] = y_data[j] + ((h*h)/2) * koeficijent * x[index_akc+1]
                   # z_data[j] = z_data[j] + ((h*h)/2) * koeficijent * x[index_akc+2]
                    index_akc = index_akc + 3 * N
                    neparni_broj = neparni_broj + 2
                    koeficijent = 2*k_pocetni - neparni_broj
            k_pocetni = k_pocetni + 1    
       
        #ax.plot(x_data, y_data, z_data) 
        
        #plt.pause(1)  
        listaTocki.append(x_data)
        listaTocki.append(y_data)
       # listaTocki.append(z_data)
        rijecnik[i] = listaTocki
        
        
        #idemo na sljedecu trajektoriju
        index = index + 3
        indeX_akcPOCETNI = indeX_akcPOCETNI + 3
    #plt.show() 

    xevi = []
    yoni = []
    zeovi = []
    x_plot = {}
    y_plot = {}
    z_plot= {}
    plt.pause(2) 
    #pocetna.remove()
    for i in range(0, k):
        
        listaPomocX = []
        listaPomocY = []
        listaPomocZ = []
        
        for j in range(0, N):
            listaPomocX = []
            listaPomocY = []
            listaPomocZ = []
            lista = rijecnik[j] 
           
            x_data = lista[0]
            y_data = lista[1]
            #z_data = lista[2]
            
            xevi.append(x_data[i])
            yoni.append(y_data[i])
            #zeovi.append(z_data[i])
            if (i != 0):
                listaPomocX = x_plot[j]
                listaPomocY = y_plot[j]
                #listaPomocZ = z_plot[j]
            listaPomocX.append(x_data[i])
            listaPomocY.append(y_data[i])
            #listaPomocZ.append(z_data[i])
            x_plot[j] =  listaPomocX
            y_plot[j] =  listaPomocY
            #z_plot[j] =  listaPomocZ

        

        #ax.scatter(xevi, yoni, zeovi) 
        
        if (i != 0):
            for l in range(0, N):
                x = []
                y = []
                z = []
                x_ = x_plot[l]
                y_ = y_plot[l]
                #z_ = z_plot[l]
                for m in range(0, len(x_)):
                    x.append(x_[m])
                    y.append(y_[m])
                   # z.append(z_[m])
                
                ax.plot(x,y)
                
                
            
        plt.pause(0.1) 
       # plt.show()  
       # plt.clf()
        #plt.close()
        #ax.set_xlabel("X vrijednosti")
        #ax.set_ylabel("Y vrijednosti")
    
    
    #plt.show()  
    #plt.clf()
    #plt.cla()
    #plt.close()
    index = 0
    for i in range(0 , N):
        ax.scatter(finalnePozicije[index], finalnePozicije[index+1], marker="s")
        index = index + 3
    plt.show()





def checkInitialValues(arrayOfPositions, pcl, pcu):
    check = False
    for i in range (0, len(arrayOfPositions)):
        pose = arrayOfPositions[i]
        if (pose[0] >= pcl[0] and pose[0] <= pcu[0] and pose[1]>=pcl[1] and pose[1]<= pcu[1] and pose[2] >= pcl[2] and pose[2] <= pcu[2]):
            check = True
        else:
            check = False
            break
    return check

def printMatrix(mat):
    np.set_printoptions(threshold=np.inf)
    print(mat)

def flatten(t):
    return [item for sublist in t for item in sublist]

def printMatrix2(mat):
    np.set_printoptions(threshold=np.inf)
    i = 0
    string = ""
    for i in range(0, mat.size ):
        string = string + ", " + str(mat[i])
        if ((i + 1) % 3 == 0):
            print(string)
            string = ""
            

def calculateAllMatrix(N, h, k, R, prostorLower, prostorUpper, pocetneBrzine):
    A_eq=generateAeq(N, k, h)

    #beq 12N
    zavrsneBrzine = pocetneBrzine
    pocetneAkceleracije= [1, 1, 1]   #pretpostavljamo za sve isto
        
    b_eq = generateBeq(N ,k , h, pocetnePoz, finalnePozicije, pocetneBrzine, zavrsneBrzine, pocetneAkceleracije)

    M = int(23.5 * k * N - 6 * N + 0.5 * k * N * N)
    A_in =  generateAin(M, N, k, h)

    
    gornjaGranicaAkc = [10, 10, 10]
    donjaGranicaAkc = [-10, -10, -10]  #treba dodatni provjeru za granice za akcleleracije formule 13 i 14
    donjaGranicaJerk  = [-0.084,  -0.084, -0.084]
    gornjaGranicaJerk=[0.084, 0.084, 0.084]
    b_in = generateBin(M, N, k, h, R,  pocetnePoz, finalnePozicije, pocetneBrzine, donjaGranicaAkc, gornjaGranicaAkc, donjaGranicaJerk, gornjaGranicaJerk, prostorLower, prostorUpper)

    return M, A_eq, b_eq, A_in, b_in




         


if __name__ == "__main__":
    start_time = time.time()
    #1 inicijalizacija
    
    N = 0  #broj letjelica, inicijalno 0
    epsilon = 0.002
    uvjet = False
    #velicina prostora za kretnju letjelica
    #OGRANICENJA PROSTORA
    prostorLower = (-30, -30, -30)
    prostorUpper = (30, 30, 30)

    #pocetne pozicije letjelica  (kasnije bi se ovo trebalo samo generirati u odnosu na neku referentu tocku jer broj letjelica moze biti veci)
    pocetnePoz=[]
    pocetnePozicije=[]  #sve pozicije
    with open("pocetnePozicije.txt") as fp:
        while True:
        
            line = fp.readline()
            line = line.strip('\n')
            if not line:
                break
        
            array = line.split(", ")
        
            point = (float(array[0]), float(array[1]), float(array[2]))
            pocetnePoz.append(point)
            pocetnePozicije.append(point)

    
    pocetnePoz = flatten(pocetnePoz)
    

    #krajnje pozicije letjelica 
    finalnePozicije=[]

    with open("krajnjePozicije.txt") as fp:
        while True:
            
            line = fp.readline()
            line = line.strip('\n')
            if not line:
                break
        
            array = line.split(", ")
        
            point = (float(array[0]), float(array[1]), float(array[2]))
            finalnePozicije.append(point)
            pocetnePozicije.append(point)

            N = N + 1
 
    
    finalnePozicije = flatten(finalnePozicije)


    #provjera zadovoljavaju li pocetne poziicje granice

    satisf = checkInitialValues(pocetnePozicije, prostorLower, prostorUpper)

    if (satisf == False):
        print("Inicijalne vrijednosti ne zadovoljavaju uvjete!")
    else:
        print("Inicijalne vrijednosti zadovoljavaju uvjete.")
        t = 3
        h = 0.02
        cal=calculateK(t, h)
        k = cal[0]
        h = cal[1]

        
        #Aeq 12Nx3NK 

        R = 0.5
        pocetneBrzine = generateMatrix(N * 3, 1)  #3 za svaki smjer
        pocetneBrzine = flatten(pocetneBrzine)
        M, A_eq, b_eq, A_in, b_in = calculateAllMatrix(N, h, k, R, prostorLower, prostorUpper, pocetneBrzine)
        
        #2 SCP
        #fesable and objective sol (ne nuzno optimal ako najde optimal super)
        uvjet = False
        i = 0
        
        g = 9.81
        while(uvjet == False):
            val = optimize(A_eq, b_eq, A_in, b_in, N, k, t, g, False)
            x = val[0]
            f_novi = val[1]
            status = val[2]

            if(status == 4 and i == 0):
                print("Nema rješenja za ovakav problem")
                break
  
            if (status == 2):
                k_stari = k
                h_stari = h
                x_stari = x
            
            
            #nadopunjava ogranicenja 17, u prvoj iteraciji tih ogranicenja nema
            val2 = nadopuni(A_in, b_in, h,  N, k, x, R, M, pocetnePoz, pocetneBrzine)
            A_in = val2[0]
            b_in = val2[1]
            
            #staus 2 znaci da je optimal, zelimo bar 2 iteracije jer u iteraciji koja je druga namjestamo ogranicenje 17
          
            if ((i + 1) % 2 == 0 and status == 2):
                h = h + 0.01
                cal=calculateK(t, h)
                k = cal[0]
                h = cal[1]

                M, A_eq, b_eq, A_in, b_in = calculateAllMatrix(N, h, k, R, prostorLower, prostorUpper, pocetneBrzine)
            
                
            

            
            if (status == 4 or status == 3 and i > 0):
                print("GOTOVO")
                #print(k_stari)
                #print(h_stari)
                k = k_stari
                h = h_stari
                x = x_stari
                uvjet = True
                break
 
            i = i + 1
    
        if (uvjet == True):
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Za zadane parametre")
            print("k: ", k)
            print("h: ", h)
            print("T: ", t )
            #printMatrix2(x)
            #print(max(x))
            #print(min(x))
            dvaDcrtanje(x, N, k, h, pocetnePoz, pocetneBrzine, finalnePozicije)
            nacrtaj(x, N, k, h, pocetnePoz, pocetneBrzine)
            