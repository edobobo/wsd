Gold standards in 18 Languages and silver training data in 15 languages.

# Format
Each language has two files for testing and for developing, i.e., 
`test-[lang_code].data.xml` and `test-[lang_code].gold.key.txt` and 
`dev-[lang_code].data.xml` and `dev-[lang_code].gold.key.txt`, respectively.
The `.data.xml` files contain sentences where each token is enclosed in a `<wf>` or `<instance>`
depending if it has to be disambiguated or not. `<instance>` tokens have an id which can be found in
`.gold.key.txt` files. These latter are lists of instance ids and gold answers (expressed as BabelNet offsets).

Here there is an excerpt of the English test:

test-en/test-en.data.xml
```xml
<?xml version='1.0' encoding='UTF-8'?>
<corpus lang="en" source="senseval2_senseval3_semeval2010_semeval2013_semeval2015">
    <text id="senseval2.d000">
        <sentence id="senseval2.d000.s000">
            <wf lemma="the" pos="DET">The</wf>
            <instance id="senseval2.d000.s000.t000" lemma="art" pos="NOUN">art</instance>
            <wf lemma="of" pos="ADP">of</wf>
            <instance id="senseval2.d000.s000.t001" lemma="change_ringing" pos="NOUN">change-ringing</instance>
            <wf lemma="be" pos="VERB">is</wf>
            <instance id="senseval2.d000.s000.t002" lemma="peculiar" pos="ADJ">peculiar</instance>
            <wf lemma="to" pos="PRT">to</wf>
            <wf lemma="the" pos="DET">the</wf>
            <instance id="senseval2.d000.s000.t003" lemma="english" pos="NOUN">English</instance>
            <wf lemma="," pos=".">,</wf>
            <wf lemma="and" pos="CONJ">and</wf>
            <wf lemma="," pos=".">,</wf>
            <wf lemma="like" pos="ADP">like</wf>
            <instance id="senseval2.d000.s000.t004" lemma="most" pos="ADJ">most</instance>
            <instance id="senseval2.d000.s000.t005" lemma="english" pos="ADJ">English</instance>
            <instance id="senseval2.d000.s000.t006" lemma="peculiarity" pos="NOUN">peculiarities</instance>
            <wf lemma="," pos=".">,</wf>
            <instance id="senseval2.d000.s000.t007" lemma="unintelligible" pos="ADJ">unintelligible</instance>
            <wf lemma="to" pos="PRT">to</wf>
            <wf lemma="the" pos="DET">the</wf>
            <instance id="senseval2.d000.s000.t008" lemma="rest" pos="NOUN">rest</instance>
            <wf lemma="of" pos="ADP">of</wf>
            <wf lemma="the" pos="DET">the</wf>
            <instance id="senseval2.d000.s000.t009" lemma="world" pos="NOUN">world</instance>
            <wf lemma="." pos=".">.</wf>
        </sentence>
...
</text>
</corpus>
```  
test-en/test-en.gold.key.txt
```text
senseval2.d000.s000.t000 bn:00005928n
senseval2.d000.s000.t001 bn:00017671n
senseval2.d000.s000.t002 bn:00108295a bn:00108382a
senseval2.d000.s000.t003 bn:00030863n
senseval2.d000.s000.t004 bn:00106953a
senseval2.d000.s000.t005 bn:00102248a
senseval2.d000.s000.t006 bn:00027769n bn:00027768n
senseval2.d000.s000.t007 bn:00107878a
senseval2.d000.s000.t008 bn:00008047n
senseval2.d000.s000.t009 bn:00063584n
```
