from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt


class AnswerFormat(BaseModel):
    answer: str
    verdict: int


question_answer_parser = RagasoutputParser(pydantic_object=AnswerFormat)


reasoning_question_prompt = Prompt(
    name="reasoning_question",
    instruction="""与えられた質問を複雑化するために、提供されたコンテキストに基づいて質問をマルチホップ推論質問に書き換えてください。
    質問に答えるためには、読者が与えられたコンテキストを使用して複数の論理的なつながりや推論を行う必要があります。
    書き換えの際に従うべきルール：
    1. 書き換えた質問が、コンテキストに存在する情報から完全に答えられることを確認してください。
    2. 質問には15語以上を含めないでください。可能な限り略語を使用してください。
    3. 質問が明確であいまいでないことを確認してください。
    4. 「提供されたコンテキストに基づいて」や「コンテキストによると」といったフレーズを質問に含めないでください。""",
    examples=[
        {
            "question": "フランスの首都はどこですか？",
            "context": "フランスは西ヨーロッパにある国です。パリ、リヨン、マルセイユなどいくつかの都市があります。パリはエッフェル塔やルーブル美術館などの文化的ランドマークだけでなく、行政の中心地としても知られています。",
            "output": "エッフェル塔と行政の中心を結びつける都市はどこですか？",
        },
        {
            "question": "Pythonでappend()メソッドは何をしますか？",
            "context": "Pythonでは、リストは単一の変数に複数の項目を格納するために使用されます。リストはデータのコレクションを格納するための4つの組み込みデータ型の1つです。append()メソッドはリストの最後に単一の項目を追加します。",
            "output": "リストが変数のコレクションを表す場合、どのメソッドが1つの項目でそれを拡張しますか？",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="english",
)


multi_context_question_prompt = Prompt(
    name="multi_context_question",
    instruction="""
    与えられた質問を再構成し複雑化するタスクです。質問に答えるには、context1とcontext2の両方から得られる情報が必要です。 
    質問を書き換える際には、以下のルールに従ってください。
        1. 書き換えた質問は非常に長くならないようにしてください。可能な限り略語を使用してください。
        2. 書き換えた質問は合理的であり、人間が理解し応答できるものでなければなりません。
        3. 書き換えた質問は、context1およびcontext2に含まれる情報から完全に答えられるものでなければなりません。 
        4. 両方のcontextを読み理解し、回答には両方のcontextからの洞察が必要となるように質問を書き換えてください。
        5. 「提供されたコンテキストに基づいて」や「コンテキストによると」といったフレーズを質問に含めないでください。""",
    examples=[
        {
            "question": "植物を緑にするプロセスは何ですか？",
            "context1": "クロロフィルは植物に緑色を与える色素で、光合成を助けます。",
            "context2": "植物の光合成は、通常、葉に集中している葉緑体で行われます。",
            "output": "緑色の原因となる色素がエネルギー生産を促進する植物の構造はどこですか？",
        },
        {
            "question": "長方形の面積をどのように計算しますか？",
            "context1": "図形の面積は、図形の寸法に基づいて計算されます。長方形の場合、これは長さと幅を掛け合わせることを含みます。",
            "context2": "長方形には4つの辺があり、対辺は等しい長さです。長方形は四辺形の一種です。",
            "output": "等しい対辺を掛け合わせると、どの四辺形の面積が求まりますか？",
        },
    ],
    input_keys=["question", "context1", "context2"],
    output_key="output",
    output_type="str",
    language="english",
)

conditional_question_prompt = Prompt(
    name="conditional_question",
    instruction="""提供された質問を、条件要素を導入することで複雑さを増すように書き換えてください。
    目標は、質問の文脈に影響を与えるシナリオや条件を取り入れることで、質問をより複雑にすることです。
    質問を書き換える際には、以下のルールに従ってください。
        1. 書き換えた質問は25語を超えてはいけません。可能な限り略語を使用してください。
        2. 書き換えた質問は合理的であり、人間が理解し応答できるものでなければなりません。
        3. 書き換えた質問は、コンテキストに存在する情報から完全に答えられるものでなければなりません。
        4. 「提供されたコンテキスト」や「コンテキストによると」といったフレーズを質問に含めないでください。""",
    examples=[
        {
            "question": "植物の根の機能は何ですか？",
            "context": "植物の根は土壌から水と栄養を吸収し、植物を地面に固定し、食物を蓄えます。",
            "output": "植物の根は、土壌の栄養分と安定性に関してどのような二重の目的を果たしますか？",
        },
        {
            "question": "ワクチンはどのようにして病気から守るのですか？",
            "context": "ワクチンは、体の免疫反応を刺激して抗体を生成し、病原体を認識して戦うことで病気から守ります。",
            "output": "ワクチンはどのように体の免疫システムを利用して病原体から守りますか？",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="english",
)

compress_question_prompt = Prompt(
    name="compress_question",
    instruction="""以下の質問を、より間接的で短くなるように書き換えてください。ただし、元の質問の本質は維持してください。
    目標は、同じ意味を伝えつつ、より直接的でない質問を作成することです。書き換えた質問は短くする必要があるため、可能な限り略語を使用してください。""",
    examples=[
        {
            "question": "地球と月の距離はどれくらいですか？",
            "output": "月は地球からどれくらい離れていますか？",
        },
        {
            "question": "チョコレートケーキを焼くのに必要な材料は何ですか？",
            "output": "チョコケーキに何が必要ですか？",
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="str",
    language="english",
)

conversational_question_prompt = Prompt(
    name="conversation_question",
    instruction="""与えられた質問を、会話の一部として2つの別々の質問に書き換えてください。それぞれの質問は、元の質問に関連する特定の側面またはサブトピックに焦点を当てるべきです。
    質問を書き換える際には、以下のルールに従ってください。
        1. 書き換えた質問は25語を超えてはいけません。可能な限り略語を使用してください。
        2. 書き換えた質問は合理的であり、人間が理解し応答できるものでなければなりません。
        3. 書き換えた質問は、コンテキストに存在する情報から完全に答えられるものでなければなりません。
        4. 「提供されたコンテキスト」や「コンテキストによると」といったフレーズを質問に含めないでください。""",
    examples=[
        {
            "question": "リモートワークの利点と欠点は何ですか？",
            "output": {
                "first_question": "リモートワークの利点は何ですか？",
                "second_question": "一方で、リモートワークの課題は何ですか？",
            },
        }
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)

question_answer_prompt = Prompt(
    name="answer_formulate",
    instruction="""与えられたコンテキストの情報を使用して質問に答えてください。回答がコンテキストに存在する場合は '1'、存在しない場合は '-1' として出力してください。""",
    output_format_instruction=get_json_format_instructions(AnswerFormat),
    examples=[
        {
            "context": """気候変動は人間の活動、特に化石燃料の燃焼による温室効果ガスの排出によって大きく影響されます。大気中の温室効果ガスの濃度が増加すると、より多くの熱が閉じ込められ、地球温暖化と天候パターンの変化を引き起こします。""",
            "question": "人間の活動はどのようにして気候変動に寄与していますか？",
            "answer": AnswerFormat.parse_obj(
                {
                    "answer": "人間の活動は主に化石燃料の燃焼による温室効果ガスの排出を通じて気候変動に寄与します。これらの排出物は大気中の温室効果ガスの濃度を高め、より多くの熱を閉じ込め、地球温暖化と気象パターンの変化を引き起こします。",
                    "verdict": "1",
                }
            ).dict(),
        },
        {
            "context": """人工知能（AI）の概念は時代とともに進化してきましたが、基本的には人間の認知機能を模倣するように設計された機械を指します。AIは学習、推論、知覚、そして場合によっては人間のように反応することができ、医療から自動運転車に至るまでの分野で重要な役割を果たしています。""",
            "question": "人工知能の主な能力は何ですか？",
            "answer": AnswerFormat.parse_obj(
                {
                    "answer": "人工知能は人間の認知機能を模倣するように設計されており、主な能力には学習、推論、知覚、そして環境に対する反応が含まれます。これらの能力により、AIは医療や自動運転などのさまざまな分野で重要な役割を果たしています。",
                    "verdict": "1",
                }
            ).dict(),
        },
        {
            "context": """ジェーン・オースティンの小説「高慢と偏見」は、エリザベス・ベネットと彼女の家族を中心に展開します。物語は19世紀のイギリスの田舎を舞台にしており、結婚、道徳、誤解の問題を扱っています。""",
            "question": "『高慢と偏見』は何年に出版されましたか？",
            "answer": AnswerFormat.parse_obj(
                {
                    "answer": "質問に対する答えはコンテキストに存在しません。",
                    "verdict": "-1",
                }
            ).dict(),
        },
    ],
    input_keys=["context", "question"],
    output_key="answer",
    output_type="json",
    language="english",
)

keyphrase_extraction_prompt = Prompt(
    name="keyphrase_extraction",
    instruction="提供されたテキストから、最も重要で特徴的な側面に焦点を当てた上位3〜5のキーフレーズを抽出してください。",
    examples=[
        {
            "text": "ブラックホールは、重力が非常に強く、光や他の電磁波を含む何もそれから逃れるエネルギーを持たない時空の領域です。一般相対性理論は、十分にコンパクトな質量が時空を変形させてブラックホールを形成できることを予測しています。",
            "output": {
                "keyphrases": [
                    "ブラックホール",
                    "時空の領域",
                    "強い重力",
                    "光と電磁波",
                    "一般相対性理論",
                ]
            },
        },
        {
            "text": "万里の長城は、約500年前に建てられた中国北部に位置する古代の一連の壁と要塞です。この巨大な壁は13,000マイル以上にわたり、古代中国の技術者たちの技術と忍耐の証です。",
            "output": {
                "keyphrases": [
                    "万里の長城",
                    "古代の要塞",
                    "中国北部",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
)

seed_question_prompt = Prompt(
    name="seed_question",
    instruction="与えられたコンテキストから完全に回答できる質問を作成してください。質問はトピックを使用して形成する必要があります。",
    examples=[
        {
            "context": "光合成は、植物が光エネルギーを化学エネルギーに変換し、クロロフィルや他の色素を使用して光を吸収するプロセスです。このプロセスは、植物の成長や酸素の生成にとって重要です。",
            "keyphrase": "光合成",
            "question": "光合成は植物の成長にどのような役割を果たしますか？",
        },
        {
            "context": "18世紀に始まった産業革命は、工場の発展と都市化をもたらし、歴史の大きな転換点となりました。",
            "keyphrase": "産業革命",
            "question": "産業革命はどのようにして歴史の大きな転換点となりましたか？",
        },
        {
            "context": "蒸発のプロセスは、水を液体から蒸気に変え、それを大気中に上昇させることで、水循環において重要な役割を果たします。",
            "keyphrase": "蒸発",
            "question": "蒸発は水循環においてなぜ重要ですか？",
        },
    ],
    input_keys=["context", "keyphrase"],
    output_key="question",
    output_type="str",
)


main_topic_extraction_prompt = Prompt(
    name="main_topic_extraction",
    instruction="与えられたテキストで詳しく説明されている2つの主要なトピックを特定して抽出してください。",
    examples=[
        {
            "text": "ブロックチェーン技術は、データ取引の整合性と透明性を確保する分散型台帳を提供します。これは、ビットコインのような暗号通貨を支える技術であり、すべての取引の安全で変更不可能な記録を提供します。金融以外では、ブロックチェーンにはサプライチェーン管理における潜在的な応用があります。これにより、業務の効率化、トレーサビリティの向上、および不正防止が可能になります。また、商品をリアルタイムで追跡し、参加者間でデータを透明に共有することができます。",
            "output": {
                "topics": [
                    "ブロックチェーン技術と暗号通貨における基盤的役割",
                    "ブロックチェーンのサプライチェーン管理への応用",
                ]
            },
        },
        {
            "text": "遠隔医療は、特に農村部やサービスが不足している地域において、医療の提供方法を革新しました。これにより、患者はビデオ会議を通じて医師と相談することができ、ケアへのアクセスを改善し、移動の必要性を減らすことができます。医療におけるもう一つの重要な進歩は、個々の遺伝子プロファイルに合わせて治療を調整する精密医療です。このアプローチにより、特定の癌や慢性疾患などのさまざまな状態に対してより効果的な治療法が実現されています。",
            "output": {
                "topics": [
                    "遠隔医療と医療アクセスへの影響",
                    "精密医療と遺伝子プロファイルに合わせた治療の役割",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
)


find_relevant_context_prompt = Prompt(
    name="find_relevant_context",
    instruction="質問と一連のコンテキストが与えられた場合、質問に答えるために最も関連性の高いコンテキストを見つけてください。",
    examples=[
        {
            "question": "フランスの首都はどこですか？",
            "contexts": [
                "1. フランスは西ヨーロッパにある国です。パリ、リヨン、マルセイユなどいくつかの都市があります。パリはエッフェル塔やルーブル美術館などの文化的ランドマークだけでなく、行政の中心地としても知られています。",
                "2. フランスの首都はパリです。また、パリはフランスで最も人口の多い都市で、人口は200万人を超えています。パリはエッフェル塔やルーブル美術館などの文化的ランドマークで知られています。",
                "3. パリはフランスの首都です。また、パリはフランスで最も人口の多い都市で、人口は200万人を超えています。パリはエッフェル塔やルーブル美術館などの文化的ランドマークで知られています。",
            ],
            "output": {
                "relevant_contexts": [1, 2],
            },
        },
        {
            "question": "カフェインは体にどのように影響し、一般的な供給源は何ですか？",
            "contexts": [
                "1. カフェインは中枢神経系の興奮剤です。一時的に眠気を防ぎ、注意力を回復させることができます。主に脳に影響を与え、神経伝達物質の機能を変えます。",
                "2. 定期的な運動は健康を維持するために不可欠です。体重をコントロールし、健康状態を改善し、エネルギーを増やし、より良い睡眠を促進することができます。",
                "3. カフェインの一般的な供給源には、コーヒー、紅茶、コーラ、エナジードリンクがあります。これらの飲料は世界中で消費されており、即座にエネルギーを供給することで知られています。",
            ],
            "output": {"relevant_contexts": [1, 2]},
        },
    ],
    input_keys=["question", "contexts"],
    output_key="output",
    output_type="json",
    language="english",
)


question_rewrite_prompt = Prompt(
    name="rewrite_question",
    instruction="""与えられたコンテキスト、質問、およびフィードバックを考慮して、質問を明確にし、回答しやすくなるように書き直してください。""",
    examples=[
        {
            "context": "エッフェル塔は鉄を使用して建設され、元々は1889年にパリで開催された万国博覧会の一時的な展示物として意図されていました。その初期の一時的な目的にもかかわらず、エッフェル塔はすぐにパリの独創性の象徴となり、都市の象徴的なランドマークとなって毎年何百万人もの訪問者を引き寄せています。ギュスターヴ・エッフェルによって設計されたこの塔のデザインは、当初は一部のフランスの芸術家や知識人から批判されましたが、その後、構造工学と建築デザインの傑作として称賛されています。",
            "question": "誰がタワーのデザインを作成しましたか？",
            "feedback": "質問は「タワーのデザインの作成者」について尋ねていますが、どの塔を指しているのか明確にしていません。世界中には多くの塔があり、具体的な塔を指定しなければ、質問は不明瞭で答えられません。質問を改善するには、対象となる特定の塔の名前または明確な説明を含める必要があります。",
            "output": "エッフェル塔のデザインを作成したのは誰ですか？",
        },
        {
            "context": "『ニューラルネットワークにおけるゼロショット学習の探求』は2021年にスミスとリーによって発表され、人工知能におけるゼロショット学習技術の応用に焦点を当てています。",
            "question": "この研究ではゼロショット評価のためにどのデータセットが使用されましたか？",
            "feedback": "質問は「この研究でゼロショット評価に使用されたデータセット」について尋ねていますが、具体的な研究についての詳細を提供していないため、研究が何を指しているのか不明確です。特定の研究にアクセスできない、またはその研究についての知識がない人には質問が不明瞭です。質問の明確さと回答可能性を向上させるためには、質問が参照する研究を明記するか、研究に関する十分なコンテキストを提供して、質問が独立して理解され回答されるようにする必要があります。",
            "output": "『ニューラルネットワークにおけるゼロショット学習の探求』の論文でゼロショット評価に使用されたデータセットは何ですか？",
        },
    ],
    input_keys=["context", "question", "feedback"],
    output_key="output",
    output_type="str",
    language="english",
)

### Filters


class ContextScoring(BaseModel):
    clarity: int
    depth: int
    structure: int
    relevance: int


class QuestionFilter(BaseModel):
    feedback: str
    verdict: int


class EvolutionElimination(BaseModel):
    reason: str
    verdict: int


context_scoring_parser = RagasoutputParser(pydantic_object=ContextScoring)
question_filter_parser = RagasoutputParser(pydantic_object=QuestionFilter)
evolution_elimination_parser = RagasoutputParser(pydantic_object=EvolutionElimination)

context_scoring_prompt = Prompt(
    name="score_context",
    instruction="""
    コンテキストが与えられた場合、次のタスクを実行し、答えを有効なJSON形式で出力してください: 提供されたコンテキストを評価し、以下の各基準に対して1（低）、2（中）、または3（高）の数値スコアをJSONレスポンスに割り当ててください：
clarity（明確さ）: 提示された情報の正確さと理解しやすさを評価します。高いスコア（3）は、情報が正確で理解しやすいコンテキストに対して予約されています。低いスコア（1）は、情報があいまいで理解しにくいコンテキストに対して与えられます。
depth（深さ）: コンテキスト内での詳細な検討と革新的な洞察の包含のレベルを判断します。高いスコアは包括的で洞察に富んだ分析を示し、低いスコアはトピックの表面的な取り扱いを示します。
structure（構造）: コンテンツがどれだけよく組織されていて、論理的に進行するかを評価します。高いスコアは、首尾一貫した組織と論理的な進行を示すコンテキストに与えられ、低いスコアは進行の構造や明確さの欠如を示します。
relevance（関連性）: 主題に対するコンテンツの適切性を判断し、無駄な脱線なしに主題に密接に焦点を当てたコンテキストに高いスコアを与え、不要な情報が散らばっているコンテキストには低いスコアを与えます。
これらの基準をキーとして、それぞれのスコアを値として反映するようにJSON出力を構成してください。
    """,
    output_format_instruction=get_json_format_instructions(ContextScoring),
    examples=[
        {
            "context": "ピタゴラスの定理は幾何学の基本原理です。これは直角三角形において、斜辺（直角に対する辺）の長さの二乗は他の2辺の長さの二乗の和に等しいと述べています。これは、cが斜辺の長さを表し、aとbが他の2辺の長さを表すとき、a^2 + b^2 = c^2と書くことができます。",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 1, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "アルベルト・アインシュタイン（1879年3月14日 - 1955年4月18日）は、ドイツ生まれの理論物理学者で、史上最も偉大で影響力のある科学者の一人と広く見なされています。",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 2, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "私はチョコレートが大好きです。本当においしいです。ところで、地球は太陽を周回していて、その逆ではありません。また、私の好きな色は青です。",
            "output": ContextScoring.parse_obj(
                {"clarity": 2, "depth": 1, "structure": 1, "relevance": 1}
            ).dict(),
        },
    ],
    input_keys=["context"],
    output_key="output",
    output_type="json",
    language="english",
)


filter_question_prompt = Prompt(
    name="filter_question",
    instruction="""質問が十分なドメイン知識を持つ場合に、明確さと回答可能性を評価し、以下の基準に基づいて評価してください：
1.独立性: 質問が追加のコンテキストや質問自体に含まれていない外部参照にアクセスする必要なく、理解され回答できるか？質問は自己完結型であるべきで、特定の文書、表、または共有されていない事前知識に依存しないことが必要です。
2.明確な意図: 質問が求めている回答または情報の種類は明確ですか？質問はその目的を曖昧にすることなく伝え、直接的で関連性のある応答を可能にするべきです。
これらの基準に基づいて、質問が具体的で、独立していて、明確な意図を持ち、提供された詳細に基づいて理解し回答可能である場合は「1」との評決を割り当ててください。曖昧さ、不十分な外部参照依存、または意図の曖昧さのためにこれらの基準の1つ以上を満たさない場合は「0」と割り当ててください。
質問が不明確であると判断された場合は、フィードバックと評決をJSON形式で提供し、改善のための提案を含めてください。質問の明確さまたはその欠如に寄与する側面を強調し、より良い理解と回答可能性のためにどのように再構成または詳細にすることができるかについてのアドバイスを提供してください。
""",
    output_format_instruction=get_json_format_instructions(QuestionFilter),
    examples=[
        {
            "question": "宇宙に関する発見とは何ですか？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "質問は「宇宙に関する発見」について尋ねていますが、特定の側面、期間、または関心のあるコンテキストを指定せずに、あまりにも曖昧で広範囲です。これは、新しい天体の発見から宇宙旅行技術の進展まで、多岐にわたるトピックを指す可能性があります。明確さと回答可能性を向上させるためには、発見の種類（例：天文学的、技術的）、期間（例：最近、歴史的）、またはコンテキスト（例：特定の研究や宇宙ミッション内）を指定することができます。",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "WMT'23研究でALMA-13B-Rはcontext1とcontext2の結果に基づいて他の翻訳モデルと比較してどうですか？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "この質問は、ALMA-13B-RモデルのWMT'23研究における他の翻訳モデルとの比較を求めていますが、「context1」と「context2」の内容を説明せず、それらにアクセスし理解することを前提としています。これは、WMT'23研究またはこれらの特定のコンテキストを知らない人にとって不明確です。広い観客向けに明確さと回答可能性を向上させるために、「context1」と「context2」を定義または説明するか、これらのコンテキストで使用された比較基準を説明すると良いでしょう。",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "KIWI-XXLとXCOMETは表1の評価スコア、翻訳モデルのパフォーマンス、および参照を超える成功率に関してゴールドスタンダード参照とどう比較されますか？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "質問は、KIWI-XXLとXCOMETモデルを「表1」におけるゴールドスタンダード参照と比較し、評価スコア、翻訳モデルのパフォーマンス、および参照を超える成功率に焦点を当てています。モデルと比較基準を明確に指定しており、その意図は明確です。しかし、質問は「表1」へのアクセスを前提としており、その内容やコンテキストを提供していないため、ソース資料に直接アクセスできない人には不明確です。一般的な読者向けに質問を明確かつ回答可能にするためには、「表1」の内容または主な調査結果を簡単に説明するか、特定の未公開文書に依存しないように質問を構成する必要があります。",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "OpenMoEにおけるUL2トレーニング目標の構成とそれが事前トレーニングに適している理由は何ですか？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "この質問は、OpenMoEフレームワーク内でのUL2トレーニング目標の構成と、その事前トレーニングに適している理由を尋ねています。質問は、関心のあるトピック（UL2トレーニング目標、OpenMoE）を明確にし、構成とその有効性の理由についての詳細な情報を求めています。しかし、特定の用語やOpenMoEとUL2の文脈を知らない人には難しいかもしれません。より広い明確さと回答可能性のためには、質問がOpenMoEとUL2トレーニング目標についての簡単な説明や文脈を含めるか、事前トレーニングの有効性の側面（例：効率性、精度、一般化）を明確にすると良いでしょう。",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question": "提供されたコンテキストに基づいて、OpenMoEのUL2トレーニング目標の詳細な構成は何ですか？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "質問は、提供されたコンテキストに基づいて、OpenMoEフレームワーク内でのUL2トレーニング目標の詳細な構成を求めていますが、このコンテキストが実際に含まれていないか、質問内で説明されていないため、質問が不明確です。質問を明確かつ回答可能にするには、関連するコンテキストを質問内に直接含めるか、外部情報を必要としないように質問を構成する必要があります。関心のある構成の具体的な側面（例：損失関数、データ拡張技術）を詳述することも、クエリを明確にするのに役立ちます。",
                    "verdict": 0,
                }
            ).dict(),
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)

evolution_elimination_prompt = Prompt(
    name="evolution_elimination",
    instruction="""与えられた2つの質問が以下の要件に基づいて等しいかどうかを確認してください：
    1. 同じ制約と要件を持っている。
    2. 問い合わせの深さと幅が同じである。
    彼らが等しい場合は1を出力し、そうでない場合は0を出力してください。""",
    output_format_instruction=get_json_format_instructions(EvolutionElimination),
    examples=[
        {
            "question1": "気候変動の主な原因は何ですか？",
            "question2": "地球温暖化に寄与する要因は何ですか？",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "どちらの質問も環境問題を扱っていますが、「気候変動」は「地球温暖化」よりも幅広い変化を含むため、問い合わせの深さが異なります。",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question1": "植物における光合成はどのように機能しますか？",
            "question2": "植物における光合成のプロセスを説明できますか？",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "両方の質問は、植物における光合成のプロセスの説明を求めており、同じ深さ、幅、回答の要件を共有しています。",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question1": "定期的な運動の健康上の利点は何ですか？",
            "question2": "健康のために定期的に運動することの利点を挙げられますか？",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "両方の質問は、定期的な運動が健康に与える良い影響についての情報を求めています。彼らは健康上の利点を挙げるための同じレベルの詳細を必要とします。",
                    "verdict": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["question1", "question2"],
    output_key="output",
    output_type="json",
    language="english",
)

testset_prompts = [
    reasoning_question_prompt,
    multi_context_question_prompt,
    conditional_question_prompt,
    compress_question_prompt,
    conversational_question_prompt,
    question_answer_prompt,
    keyphrase_extraction_prompt,
    seed_question_prompt,
    main_topic_extraction_prompt,
    find_relevant_context_prompt,
    question_rewrite_prompt,
    context_scoring_prompt,
    filter_question_prompt,
    evolution_elimination_prompt,
]
