# kaggle-NeurIPS2024
kaggle competition [NeurIPS 2024 - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA) 2024/04/04~2024/07/08<br>
参加期間: 4/19-7/8

## result
32位 銀メダル<br>
Public 0.407 Private 0.267<br>
Public 918位　→　Private 32位
### review
今回のコンペは思ったより時間をさけず、またデータの大きさのせいで取り回しに苦戦した。半ばあきらめて、shake upの可能性にかけてensembleしたsubmitで銀メダル獲得。expertになった。<br>
全体としては、shake upがとてつもなく、シンプルな1dCNNで過学習が抑えられた単一モデルが効力を発揮したっぽい。<br>
nonshareや非triazineの予測は、trainingデータからの学習ではかなり無理があったように思える。<br>
結局のところデータ勝負かつ、今回に関しては完全に運の要素が強かった。

### exp
- exp3
    - base
        - Model best validation BCE = 0.009349 on epoch 11
        - validation APS = 0.708215 on epoch 11
        - validation AUC = 0.988936 on epoch 11
        - validation APS of BRD4 = 0.603703 on epoch 11
        - validation APS of HSA = 0.402498 on epoch 11
        - validation APS of sEH = 0.902392 on epoch 11
        - Elapsed time = 23:53:40
            - node_dim=80(79), edge_dim=16(12)
            - public 0.399
            - 72,8よりは上がっているが、一番初めのモデルよりは落ちる。これはseedの問題はあるし、あんまりPLに依存せず、validationを信じた方がよさそう。ただ過学習には注意。
    - lr1e-4_layer5
        - 特徴量増やした状態で、lrを下げて、MPNNlayer増やすかつ、APS向上したときに保存するように変更。
        - Model best APS = 0.720607 on epoch 29
        - validation BCE = 0.009186 on epoch 29
        - validation AUC = 0.989698 on epoch 29
        - validation APS of BRD4 = 0.618622 on epoch 29
        - validation APS of HSA = 0.412452 on epoch 29
        - validation APS of sEH = 0.912093 on epoch 29
        - Elapsed time = 1 day, 6:46:49
        - PL 0.381
    - lr1e-4_graphdim128
        - 同様にlr下げて、graphdimあげる
        Model best APS = 0.721140 on epoch 26
        - validation BCE = 0.009018 on epoch 26
        - validation AUC = 0.989653 on epoch 26
        - validation APS of BRD4 = 0.618091 on epoch 26
        - validation APS of HSA = 0.411803 on epoch 26
        - validation APS of sEH = 0.913199 on epoch 26
        - Elapsed time = 22:49:19
        - PL 0.404
    - 二つとも最初のに勝てない。過学習している分がnonshareの部分を下げているか。
    - shareは上がっていてもいいはずなので、nonshareは別のモデルで対応する必要があるか。
    - 最終submitとしては、PLがよかった最初のモデルと、コンペの特性上ensembleが多い方がいいという判断で、exp3の結果をすべてensembleしたものを提出。