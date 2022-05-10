from fastText import  *
import fasttext.util
import numpy as np
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import warnings
import json, time
import zeyrek
import random
from random import shuffle
random.seed(1)
start_time = time.time()

fasttext.util.download_model('tr', if_exists='ignore')
# pip install xlrd
WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')
analyzer = zeyrek.MorphAnalyzer()

ft = fasttext.load_model(r'cc.tr.300.bin')
# sent = "çocuk sağlığı hastalıkları hastaneler hastanelerimiz i̇ vm medical park florya i̇eu medical park i̇zmir i̇stinye üniversitesi hastanesi bahçeşehir i̇sü medical park gaziosmanpaşa medical park ankara batıkent medical park antalya medical park bahçelievler medical park batman medical park çanakkale medical park elazığ medical park gaziantep medical park gebze medical park göztepe medical park ordu medical park tarsus medical park tokat medical park trabzon karadeniz medical park trabzon yıldızlı vm medical park ankara keçiören vm medical park bursa vm medical park kocaeli vm medical park maltepe vm medical park mersin vm medical park pendik vm medical park samsun tıbbi birimler tıbbi bölümlerimiz acil servis ağız diş sağlığı aile hekimliği alerji i̇mmünoloji algoloji anestezi reanimasyon beslenme diyet beyin sinir cerrahisi nöroşirürji biyokimya check up evde bağışıklık testi çocuk cerrahisi çocuk endokrinolojisi çocuk enfeksiyon hastalıkları çocuk gastroenteroloji hepatoloji beslenme çocuk gelişim uzmanlığı çocuk hematolojisi çocuk i̇mmünoloji alerji çocuk kalp damar cerrahisi çocuk kardiyolojisi çocuk nefrolojisi çocuk nörolojisi çocuk onkolojisi çocuk romatolojisi çocuk sağlığı hastalıkları çocuk ürolojisi çocuk ergen psikiyatrisi çocuk yoğun bakım dermatoloji cildiye diyabetik ayak polikliniği el cerrahisi endokrinoloji metabolizma hastalıkları enfeksiyon hastalıkları mikrobiyoloji fizik tedavi rehabilitasyon gastroenteroloji gastroenteroloji cerrahisi genel cerrahi genel yoğun bakım genetik hastalıklar tanı merkezi girişimsel radyoloji göğüs cerrahisi göğüs hastalıkları göz sağlığı hastalıkları havacılık tıp merkezi hematoloji i̇ç hastalıkları dahiliye i̇nme merkezi ünitesi jinekolojik onkoloji cerrahisi kadın hastalıkları doğum kalp damar cerrahisi kardiyoloji konuşma dil terapisi kulak burun boğaz laboratuvar hizmetleri medikal estetik medikal onkoloji meme cerrahisi meme sağlığı ünitesi mikrobiyoloji nefroloji neonatoloji nöroloji nükleer tıp organ nakli ortodonti ortopedi travmatoloji patoloji pedodonti perinatoloji plastik rekonstrüktif estetik cerrahi protetik diş tedavisi psikiyatri psikoloji radyasyon onkolojisi radyoloji romatoloji saç ekimi kliniği spor hekimliği tıbbi genetik tüp bebek ivf üroloji yenidoğan yoğun bakım hekimler sağlık rehberi hasta rehberi anlaşmalı kurumlar i̇letişim hızlı randevu al laboratuvar sonuçlarını görüntüle international patients hastaneler tıbbi birimler hekimler sağlık rehberi hasta rehberi hızlı randevu giriş yap üye ol doktor adı hastalık adı hastane adı hastaneler hospital hospitalname hekimler doctor doctorname bölümler branch branchname branch hospitalcount hastane sağlık rehberi content title sayfa bağlantıları menu menuname daha fazla sonuç çocuk sağlığı hastalıkları tıbbi birimler çocuk sağlığı hastalıkları pediatri olarak bilinen çocuk sağlığı hastalıkları bölümü doğumdan ergenliğe kadar uzanan süreç içinde yer alan bireylerin tanı takip tedavisi ilgilenen bilim dalıdır 18 yaş arasında çocuk olarak tanımlanan kişilerin doğumsal hastalıkları doğum sonrası düzenli olarak uygulanması gereken aşı takibi mental fiziksel motor gelişimi pediatri hekimlerince takip edilir bu süreçte yapılan rutin muayenelerde bebeklerin boy kilo beslenme benzeri gelişiminin yanı sıra günlük yaşam becerilerinin ifade anlamalarının gelişimi nörolojik psikolojik gelişimleri gibi pek çok süreç kontrol takip edilir henüz iletişim becerisi gelişmemiş bebeklerin hastalıklarının zorlu tanı tedavisi ilgilenen hekimler çocukluk döneminde bulunan bireylerin muayenelerini ise onların psikolojilerini göz önünde bulundurularak yapar çocuk sağlığı hastalıkları hekimleri yıl tıp fakültesi eğitimi aldıktan sonra yıl pediatri bölümünde ihtisas yaparlar çocuk sağlığı hastalıkları bölümünün pek çok yan dalı mevcuttur bu dallarda uzmanlaşan pediatristler ek eğitim sürecinden geçerler çocuk sağlığı hastalıkları bölümünde hizmet verdiğimiz alanlar çocuk cerrahisi çocuk cerrahları yıl tıp eğitimini tamamladıktan sonra yıl çocuk cerrahisi bölümünde uzmanlık eğitimi alırlar doğumsal anomaliler olarak bilinen doğum esnasında var olan fiziksel şekil bozuklukların pek çok hastalığın tanı tedavisi çocuk cerrahisi bölümü hekimleri ilgilenir kalp hastalıkları hariç olmak üzere göğüs jinekoloji onkoloji endokrin sindirim sistemi travma karın göbek fıtığı cerrahisi başlıca uzmanlık alanlarıdır açık ameliyatların yanı sıra kapalı ameliyat olarak bilinen laparoskopik torakoskopik cerrahi girişimler çocuk cerrahı tarafından uygulanır çocuk kardiyolojisi çocuk kardiyologları yıl tıp fakültesi eğitimi aldıktan sonra yıl pediatri alanında uzmanlık eğitimi yaparlar sonrasında yıl çocuk kalp damar cerrahisi alanında yan dal eğitimi alarak uzun bir eğitim sürecini tamamlarlar bu bilim dalı anne karnında yer alan bebekten başlayarak 18 yaşındaki çocuklara kadar olan bireylerin kalp hastalıklarının tanı tedavisi ilgilenir çocuk kardiyolojisi üfürüm gibi doğumsal olarak tanımlanan konjenital kalp rahatsızlıklarının yanı sıra edinsel yani sonradan görülen hastalıkların tanı tedavisi ilgilenir çocuklarda çoğu vakada küçük bir kesi üzerinden yapılan cerrahi müdahaleler çocukların hızlı metabolizması sayesinde iyileşme süreci yetişkinlere göre çok daha hızlı olur çocuk nörolojisi pediatrik nöroloji 18 yaş aralığında yer alan bireylerin beyin sinir sistemine bağlı hastalıkların tanı tedavisi ilgilenen bilim dalıdır anne karnında doğum esnasında meydana gelen beyin hasarlanmaları felç nörolojik gelişim bozuklukları ateşli ateşsiz havaleler otizm baş dönmesi baş ağrısı epilepsi kas hastalıkları gibi pek çok hastalık bu bölümün hekimleri ilgilenir çocuk nörolojisi hekimleri yıl tıp fakültesi yıl çocuk sağlığı hastalıkları eğitimi aldıktan sonra yıl bu bölüm üzerine yan dal eğitimi alırlar çocuk ergen psikiyatrisi fiziksel gelişimin yanı sıra psikolojik gelişimin oldukça hızlı olduğu 18 yaş altı dönemde meydana gelen hastalıkların tanı tedavisi çocukların psikososyal sosyal öz bakım ince kaba motor becerileri dil bilişsel gelişimi gibi pek çok gelişim testi dikkat testleri çocuk ergen psikiyatrisi bölümü hekimlerince değerlendirilir bireysel psikoterapi aile terapisi gibi tedavi yaklaşımları uzman hekimlerce yapılır bu bölümün hekimleri yıllık tıp fakültesi eğitimi aldıktan sonra yıl bu bölümün eğitimini tamamlar çocuk hematolojisi kan hastalıkları anlamına gelen hematoloji kan kemik iliğinin işlevi yapısı ilgilenen bilim dalıdır hâlsizlik yorgunluk sarılık polisitemi tekrarlayan enfeksiyonlar kilo problemleri kanamalar lenf bezlerinde meydana gelen büyüme kemik eklem ağrıları hemanjiom lenfoma lösemi hodgkin lenfoma gibi pek çok hastalığın yanı sıra çocukluk çağında sıklıkla görülen akdeniz nutrisyonel anemileri çocuk hematolojisi ilgilenir kemik iliği kan nakli gibi transplantasyonlar bu bölümün hekimlerince gerçekleştirilir çocuk sağlığı hastalıkları bölümünün yan dalı olan bu bölüm hekimleri yıl tıp fakültesi yıl çocuk sağlığı hastalıkları yıl çocuk hematolojisi eğitimi alırlar çocuk nefrolojisi 18 yaş aralığında bulunan bireylerin böbrek idrar yolları hastalıkları üzerine uzmanlaşan çocuk nefrologları tanı tedavinin yanı sıra hastalığın tekrar oluşumunun önüne geçerek böbreklerin tamir edilemez şekilde hasarlanmasının önüne geçmek için koruyucu tedavi uygularlar glomerüler tübülointerstisyel böbrek parankim hastalıklarının yanı sıra doğuştan gelen edinilmsel böbrek kistik hastalıklarıyla çocuk nefrolojisi bölümü hekimleri ilgilenir ailevi akdeniz ateşi lupus gibi sistemik hastalıklara bağlı olarak gelişen böbrek tutulumları bebek çocuklarda görülen idrar yolu hastalıkları bu bilim dalınca tetkik tedavi edilir bazı durumlarda anne karnında bulunan bebekte gelişen idrar yolu enfeksiyonları böbrek hastalıklarının tanı tedavisi çocuk nefrologları tarafından takip tedavi edilir hekimler yıl tıp fakültesi yıl çocuk sağlığı hastalıkları yıl çocuk nefrolojisi eğitimi alırlar çocuk i̇mmünolojisi alerji hastalıkları yıl tıp fakültesi yıl çocuk sağlığı hastalıkları yıl çocuk immünolojisi alerji hastalıkları eğitimi alan hekimler immünolojik kaynaklı hastalıkların tanı tedavisi ilgilenir astım ürtiker egzama alerjik nezle gibi pek çok hastalığın tanı tedavisinin yanı sıra gıda böcek ilaç alerjileri solunum fonksiyon testleri çocuk immünolojisi alerji hastalıkları uzmanınca takip tedavi edilir çocuk onkolojisi çocuk onkolojisi çocuk hematolojisi onkolojisi bölümü hekimlerince çocukluk çağında görülen kanserin tanı tedavisi ilgilenir sıklıkla lenfoma lösemi beyin kemik diğer organlarda bulunan tümörler ilgili farklı hastalıkların tedavisi ilgilenen çocuk onkolojisi yüksek teknoloji görüntüleme cihazlarının bulunduğu radyoloji bölümü diğer pek çok bölüm multidisipliner olarak çalışır erken tanı tedavi çocukluk çağı kanserlerinin pek çoğu tam olarak iyileşmektedir çocuk ürolojisi anne karnından 18 yaşına kadar olan kişilerin genital bölge hastalıkları böbrek idrar yollarında bulunan hastalıkların tanı tedavisi ilgilenen çocuk ürolojisi uzmanları yıl tıp fakültesi eğitimi aldıktan sonra çocuk cerrahisi üroloji ihtisası yaptıktan sonra yıl çocuk ürolojisi eğitimi alırlar anne karnındaki bebekte tespit edilen böbrek büyümesi olarak bilinen hidronefroz polikistik böbrek upj uvj darlıkları mesane enfeksiyonları peygamber sünneti olarak bilinen hipospadias inmemiş testis sünnet over kistleri labial yapışıklıklar gibi pek çok hastalığın tedavisi ilgilenirler çocuk gastroenteroloji hepatoloji beslenme 18 yaşına kadar çocukların karaciğer mide bağırsak hastalıklarının tanı tedavisinin yapıldığı bu bölümde bebek çocuklarda görülen iştahsızlık kronik karın ağrıları gastrit peptik ülser çölyak hastalığı pankreas hastalıkları karaciğer yetmezliği gibi hastalıkların tanı tedavisi yapılır gastroskopi kolonoskopi gibi görüntüleme yöntemlerinin yapıldığı bölümün hekimleri yıl çocuk gastroenterolojisi eğitimi birlikte toplam 7y ıl uzmanlık eğitimi almaktadır çocuk endokrinolojisi hormonlar hormonların salgılandığı iç salgı bezleri ilgili hastalıklarla uğraşan çocuk endokrinolojisi büyüme gelişim gerilikleri erken geç görülen ergenlik obezite adet düzensizlikleri aşırı tüylenme kemik hastalığı gibi pek çok farklı hastalığın tanı tedavisi ilgilenir hekimler yıl tıp fakültesi yıl çocuk sağlığı hastalıkları yıl çocuk endokrinolojisi eğitimi alırlar çocuk yoğun bakımı yaşam bulguları risk altında olan 28 günlük bebekler 18 yaşına kadar olan çocukların çocuk yoğun bakım ünitelerinde 365 gün 24 saat kesintisiz olarak çocuk yoğun bakım uzmanlarınca izlenir bu hastalar genellikle sistem organ yetersizliğine bağlı olarak geçirdikleri ameliyat sonrası genel durumlarının takip edilmesi için bu ünitelerde gözlem altında tutulurlar çocuk yoğun bakım hekimleri yıllık tıp fakültesini tamamladıktan sonra yıl çocuk sağlığı hastalıkları bölümünü yıl çocuk yoğun bakımı bölümünü tamamlar hızlı randevu al birim olan hastaneler sağlık rehberi i̇çerikleri hekimler çocuk sağlığı hastalıkları birimi olan hastaneler medical park bahçelievler medical park batman vm medical park bursa medical park elazığ medical park gaziantep medical park gebze medical park göztepe i̇eu medical park i̇zmir medical park ordu vm medical park samsun medical park tarsus medical park tokat medical park antalya i̇stinye üniversitesi hastanesi bahçeşehir i̇ vm medical park florya vm medical park kocaeli medical park trabzon karadeniz medical park trabzon yıldızlı medical park ankara batıkent vm medical park pendik i̇sü medical park gaziosmanpaşa vm medical park mersin vm medical park maltepe medical park çanakkale vm medical park ankara keçiören sağlık rehberi çocuk uyku bozuklukları polikliniği çocuk sağlığı hastalıkları balık yağı faydaları nelerdir çocuk sağlığı hastalıkları bebeklerde ateş kaç olmalı çocuk sağlığı hastalıkları 12 aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları 11 aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları 10 aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları aylık bebek gelişimi nasıl olur çocuk sağlığı hastalıkları kolik bebek nedir kolik bebek müzikleri nelerdir çocuk sağlığı hastalıkları bebeklerde diş çıkarma belirtileri nelerdir çocuk sağlığı hastalıkları raşitizm nedir raşitizmin belirtileri nelerdir çocuk sağlığı hastalıkları disleksi nedir belirtileri tedavi yöntemleri nelerdir çocuk sağlığı hastalıkları okul öncesi çocuklar için hangi sağlık kontrolleri yaptırılmalı çocuk sağlığı hastalıkları yaş sendromu nedir çocuk sağlığı hastalıkları kızıl hastalığı nedir belirtileri tedavi yöntemleri nelerdir çocuk sağlığı hastalıkları kawasaki hastalığı nedir çocuk sağlığı hastalıkları spina bifida nedir belirtileri tedavi yöntemleri nelerdir çocuk sağlığı hastalıkları aşı takvimi çocuk sağlığı hastalıkları prematüre bebek nedir çocuk sağlığı hastalıkları bebeklerde cilt hastalıkları çocuk sağlığı hastalıkları ay ay bebek gelişimi çocuk sağlığı hastalıkları altıncı hastalık nedir çocuk sağlığı hastalıkları bebeklerde ek gıdaya geçiş çocuk sağlığı hastalıkları çocuklarda görülen bulaşıcı hastalıklar çocuk sağlığı hastalıkları ergenlik döneminde çocuk ebeveyn ilişkileri çocuk sağlığı hastalıkları çocuklarda üst solunum yolu hastalıkları nelerdir çocuk sağlığı hastalıkları aromaterapi nedir çocuk sağlığı hastalıkları kelebek hastalığı lupus nedir belirtileri tedavi yöntemleri nelerdir çocuk sağlığı hastalıkları el ayak hastalığı nedir belirti tedavi yöntemleri nelerdir çocuk sağlığı hastalıkları en iyi dostlarımız evcil hayvanlar çocuk sağlığı hastalıkları fenilketonüri hastalığı nedir belirti tedavi yöntemleri nelerdir çocuk sağlığı hastalıkları bebeklerde çocuklarda ateş çocuk sağlığı hastalıkları krup hastalığı nedir çocuk sağlığı hastalıkları down sendromu nedir down sendromu belirtileri nelerdir çocuk sağlığı hastalıkları i̇lk yardımı siz yapın çocuk sağlığı hastalıkları boğmaca hastalığı nedir çocuk sağlığı hastalıkları yenidoğan tarama testleri çocuk sağlığı hastalıkları evde bebek bakımında dikkat edilmesi gerekenler çocuk sağlığı hastalıkları el yaralanmaları çocuk sağlığı hastalıkları bebeklerde gaz problemi çocuk sağlığı hastalıkları hekimlerimiz filtreleyin hospital name filtreyi sıfırla seçmiş olduğunuz hastane için sonuç bulunamamıştır prof dr ayşe esra yılmaz çocuk sağlığı hastalıkları medical park ankara batıkent profili gör profil hızlı randevu al randevu prof dr figen şahin dağlı çocuk sağlığı hastalıkları vm medical park pendik sosyal pediatri profili gör profil hızlı randevu al randevu prof dr ömer cevit çocuk sağlığı hastalıkları i̇sü medical park gaziosmanpaşa profili gör profil hızlı randevu al randevu doç dr ahmet sami yazar çocuk sağlığı hastalıkları vm medical park pendik profili gör profil hızlı randevu al randevu doç dr aslı aslan çocuk sağlığı hastalıkları i̇eu medical park i̇zmir profili gör profil hızlı randevu al randevu doç dr mehmet akın çocuk sağlığı hastalıkları medical park göztepe profili gör profil hızlı randevu al randevu doç dr muhammet şükrü paksu çocuk sağlığı hastalıkları vm medical park samsun profili gör profil hızlı randevu al randevu doç dr nureddin vurgun çocuk sağlığı hastalıkları vm medical park bursa profili gör profil hızlı randevu al randevu dr öğr üyesi caner yıldız çocuk sağlığı hastalıkları medical park ankara batıkent profili gör profil hızlı randevu al randevu dr öğr üyesi dilek hatipoğlu çocuk sağlığı hastalıkları i̇stinye üniversitesi hastanesi bahçeşehir profili gör profil hızlı randevu al randevu dr öğr üyesi ecmel erdağ çocuk sağlığı hastalıkları medical park gebze profili gör profil hızlı randevu al randevu dr öğr üyesi emel şen çocuk sağlığı hastalıkları vm medical park samsun profili gör profil hızlı randevu al randevu 12 devamını göster doctor doctorname doctor branchname doctor subtitle profili gör hızlı randevu al hızlı randevu al hasta rehberi ziyaretçi politikası refakatçi politikası güvenlik bilgisi politikası şikayet politikası i̇leri teknolojiler hasta hakları birimi memnuniyet sağlama süreci hasta odaları ameliyathaneler kurumsal haberler hakkımızda başkanın mesajı vizyon kalite yönetim sistemi yatırımcı i̇lişkileri sponsorluklar kişisel verilerin korunması hakkında aydınlatma metni i̇letişim online i̇şlemler hızlı randevu al laboratuvar sonuçlarını görüntüle sizi dinliyoruz online doktor görüşmesi 444 44 84 international patient information hasta rehberi ziyaretçi politikası refakatçi politikası güvenlik bilgisi politikası şikayet politikası i̇leri teknolojiler hasta hakları birimi memnuniyet sağlama süreci hasta odaları ameliyathaneler kurumsal haberler hakkımızda başkanın mesajı vizyon kalite yönetim sistemi yatırımcı i̇lişkileri sponsorluklar kişisel verilerin korunması hakkında aydınlatma metni i̇letişim online i̇şlemler hızlı randevu al laboratuvar sonuçlarını görüntüle sizi dinliyoruz online doktor görüşmesi 444 44 84 international patient information jci uluslararası akreditasyon daha fazla bilgi gizliliğiniz bizim için önemlidir 6698 sayılı kişisel verilerin korunması kanunu hakkında bilgilendirme buradan ulaşabilirsiniz copyright 1993 2022 medical park hastaneler grubu kullanım koşulları gizlilik politikası"
# sample_text = "çocuk sağlığı hastalıkları hastaneler hastanelerimizi̇ medical park floryada tedavi görüyorlar"
# words = sent.split(' ')
# words = [word for word in words if word is not '']
# num_words = len(words)

def get_fasttext_vector(words, n): # n=len(words)//70
    new_words = words.copy()
    sent_list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # sentence = words.lower()
        # tokens = WPT.tokenize(sentence)
        # new_tokens = [token for token in tokens if token not in stop_word_list]
        # wv = ft.get_sentence_vector(sentence)
        random_word_list = list(set([word for word in words if word not in stop_word_list]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for i in range(n):
            random_word = random.choice(random_word_list)
            nearest_neighbors = ft.get_nearest_neighbors(random_word, k=1)
            num_replaced += 1
            augmented_word = nearest_neighbors[0][1]
            # augmented_word = augmented_word.lower()
            new_words = [augmented_word if word == random_word else word for word in new_words]
        # for random_word in random_word_list:
        #
        #     # wv = ft.get_wor    # result = analyzer.analyze(word)
        #     #             # for word_analyse in result:
        #     #             #     for parse in word_analyse:
        #     #             #         if parse.pos == 'Verb':
        #     #             #             print(word_analyse[0].word)d_vector(word)
        #     # wv = ft.get_sentence_vector(sentence)
        #
        #
        #     nearest_neighbors = ft.get_nearest_neighbors(random_word, k=1)
        #     num_replaced += 1
        #     augmented_word = nearest_neighbors[0][1]
        #     # augmented_word = augmented_word.lower()
        #     new_words = [augmented_word if word == random_word else word for word in new_words]
        #
        #     if num_replaced >= n:  # only replace up to n words
        #         break
            # sent_list.append(augmented_sentence)
            # this is stupid but we need it, trust me

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
    return new_words

# print('ORIGINAL SENTENCE with lenght :', len(sent) , sent)
# augmented_sentence = ' '.join(get_fasttext_vector(words))
# print('\nAUGMENTED SENTENCE with lenght :',len(augmented_sentence) , augmented_sentence)
#
# from difflib import SequenceMatcher
#
# ratio = SequenceMatcher(None, sent, augmented_sentence).ratio()
# print('similarity between original and augmented sentence :', ratio)
# print('Total run-time : ', time.time()-start_time)

def doc_processing(file_path):
    with open(file_path) as f:
        docs = f.readlines()
        for doc in docs:
            label = []
            sentence = []
            for words in doc.split():
                if words.startswith("__label"):
                    label.append(words)
                else:
                    sentence.append(words)
            # original_doc = ' '.join(sentence)
            # original_doc_words = original_doc.split(' ')
            # original_doc_words = [word for word in original_doc_words if word is not '']
    return  label, sentence




n = 100 # the number of words to randomly select and find nearest neighbour
N = 5 #The number of folds you want to augment the data
with open('metin_dosyasi.txt') as f , open('augmented_metin_dosyasi.txt', 'a') as aug_f:
    docs = f.readlines()
    for doc in docs:
        label = []
        sentence = []
        for words in doc.split():
            if words.startswith("__label"):
                label.append(words)
            else:
                sentence.append(words)
    # original_doc_words = doc_processing('metin_dosyasi.txt')
        original_doc = ' '.join(sentence)
        print(' '.join(label) + ' ' + original_doc)
        aug_f.writelines(' '.join(label) + ' ' + original_doc + "\n")
        aug_start_time = time.time()
        for n_aug in range(N):
            augmented_doc = ' '.join(get_fasttext_vector(sentence, n))
            aug_f.writelines(' '.join(label) + ' ' + augmented_doc + "\n")
            # print(' '.join(label) + ' ' + augmented_doc)
        print(f'Augmentaion for {N} times run-time : ', time.time() - aug_start_time)
        # print(' '.join(label) + augmented_doc  == ' '.join(label) + original_doc)

print('Total run-time : ', time.time()-start_time)