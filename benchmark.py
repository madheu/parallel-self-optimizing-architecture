#!/usr/bin/env python3
"""
PSOA Benchmark - 推理能力测试集
=================================
20道题目：10道GSM8K风格小学数学题 + 10道逻辑谜题
"""

import json
import sys

MATH_QUESTIONS = [
    {"id": "math_01", "question": "小明有15颗糖果，他给了小红一半，然后又买了8颗。请问小明现在有多少颗糖果？", "expected": "15", "difficulty": "easy", "solution": "15的一半是7或8。给7颗剩8颗，8+8=16；给8颗剩7颗，7+8=15。标准答案为15。"},
    {"id": "math_02", "question": "一个农场有鸡和兔子共35只，脚共有94只。请问鸡和兔子各有多少只？", "expected": "鸡23只，兔12只", "difficulty": "medium", "solution": "设鸡x只，兔y只。x+y=35, 2x+4y=94。解得y=12, x=23。"},
    {"id": "math_03", "question": "一列火车以每小时60公里的速度行驶，需要多少分钟才能行驶50公里？", "expected": "50", "difficulty": "easy", "solution": "时间=距离/速度=50/60小时=50分钟。"},
    {"id": "math_04", "question": "一本书原价80元，先涨价10%，再降价10%。请问现在的价格是多少元？", "expected": "79.2", "difficulty": "medium", "solution": "涨价后80*1.1=88元。降价后88*0.9=79.2元。"},
    {"id": "math_05", "question": "甲乙两人同时从相距100公里的两地相向而行，甲每小时走6公里，乙每小时走4公里。一只狗以每小时10公里的速度在两人之间来回跑，直到两人相遇。请问狗跑了多少公里？", "expected": "100", "difficulty": "hard", "solution": "相遇时间=100/10=10小时。狗跑了10*10=100公里。"},
    {"id": "math_06", "question": "一个水池有进水管和出水管。单独开进水管8小时注满，单独开出水管12小时放空。如果两管同时开，需要多少小时注满水池？", "expected": "24", "difficulty": "hard", "solution": "净速率=1/8-1/12=1/24池/小时。需要24小时。"},
    {"id": "math_07", "question": "某商品打八折后再减100元，最终价格是1100元。请问原价是多少元？", "expected": "1500", "difficulty": "medium", "solution": "0.8x-100=1100, x=1500元。"},
    {"id": "math_08", "question": "一个正方形边长为10cm，连接各边中点得到一个新的正方形。新正方形的面积是多少平方厘米？", "expected": "50", "difficulty": "medium", "solution": "新正方形面积是原正方形的一半，100/2=50。"},
    {"id": "math_09", "question": "有100个球，编号1到100。请问其中有多少个数字1出现在所有编号中？", "expected": "21", "difficulty": "hard", "solution": "个位1出现10次，十位1出现10次，其中11算两次。总计21个。"},
    {"id": "math_10", "question": "一个圆柱体的高是10cm，底面半径是3cm。请问它的体积是多少立方厘米？（pi取3.14）", "expected": "282.6", "difficulty": "easy", "solution": "V=pi*r^2*h=3.14*9*10=282.6。"},
]

LOGIC_QUESTIONS = [
    {"id": "logic_01", "question": "三门问题（Monty Hall Problem）：你参加一个节目，面前有三扇门。一扇门后面是一辆汽车，另外两扇门后面各是一只山羊。你选择了1号门。主持人（知道门后有什么）打开了3号门，后面是一只山羊。现在主持人问你：你要换成2号门吗？换门能增加你赢得汽车的概率吗？如果换，中奖概率是多少？", "expected": "2/3", "difficulty": "hard", "solution": "换门中奖概率=2/3。初始选对概率1/3，选错概率2/3。选错时换必中，选对时换必不中。", "trap": "很多人直觉认为是1/2"},
    {"id": "logic_02", "question": "两个骰子掷出后，点数之和为7的概率是多少？", "expected": "1/6", "difficulty": "easy", "solution": "36种组合中和为7的有6种：(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)。概率=6/36=1/6。"},
    {"id": "logic_03", "question": "一个村庄里只有两种人：骑士（永远说真话）和无赖（永远说假话）。你遇到两个人A和B。A说：我们两个都是无赖。请问A和B分别是什么人？", "expected": "A是无赖，B是骑士", "difficulty": "medium", "solution": "如果A是骑士则他说真话意味着他是无赖，矛盾。所以A是无赖。A说假话意味着不都是无赖，所以B是骑士。"},
    {"id": "logic_04", "question": "你有8个外观相同的球，其中一个比其他重。你只有一个天平，最少称几次能保证找到那个重球？", "expected": "2", "difficulty": "medium", "solution": "分3-3-2三组。第一次称两组3个的。平衡则在2个组中再称1次；不平衡则在重的3个中任取2个称1次。共2次。"},
    {"id": "logic_05", "question": "一个池塘里的睡莲面积每天扩大一倍。如果60天可以覆盖整个池塘，那么覆盖半个池塘需要多少天？", "expected": "59", "difficulty": "easy", "solution": "每天翻倍，第60天全池，第59天半池。答案是59天。", "trap": "直觉可能答30"},
    {"id": "logic_06", "question": "有3顶红帽子和2顶白帽子。3个人站成一排，每人随机戴一顶。最后的人能看到前面两人，中间的人能看到最前面的人，最前面的人什么都看不到。问最后的人：你知道自己戴什么帽子吗？答：不知道。问中间的人：你知道吗？答：不知道。问最前面的人：你知道吗？", "expected": "红帽子", "difficulty": "hard", "solution": "最后的人说不知道说明前两人不都是白帽。中间的人说不知道说明最前面不是白帽（否则中间就能推断自己是红）。所以最前面的人是红帽子。"},
    {"id": "logic_07", "question": "一个农夫要把狼、羊和白菜运过河。船每次只能带一样东西。如果狼和羊单独在一起，狼会吃羊；如果羊和白菜单独在一起，羊会吃白菜。最少需要运几次（单程）才能把所有东西都安全运过去？", "expected": "7", "difficulty": "medium", "solution": "1.带羊去 2.空回 3.带狼去 4.带回羊 5.带白菜去 6.空回 7.带羊去。共7次。"},
    {"id": "logic_08", "question": "某工厂生产一批零件，如果每天多生产3个，就可以提前4天完成任务。如果每天少生产3个，就要推迟6天才能完成。请问这批零件一共有多少个？", "expected": "360", "difficulty": "hard", "solution": "(x+3)(y-4)=xy, (x-3)(y+6)=xy。解得x=15,y=24,总量=360。"},
    {"id": "logic_09", "question": "一个房间里有100个人，每个人都说了谎。第一个人说至少有1个人说了真话，第二个人说至少有2个人说了真话，...，第100个人说至少有100个人说了真话。请问有几个人说了真话？", "expected": "0", "difficulty": "medium", "solution": "题目已声明每个人都说了谎，所以0人说真话。所有陈述都是假的，与前提一致。", "trap": "题目已声明每个人都说了谎，这是关键前提"},
    {"id": "logic_10", "question": "你有一根不均匀的绳子，从头烧到尾需要恰好1小时。你用多根这样的绳子，如何测量45分钟？", "expected": "2根", "difficulty": "medium", "solution": "第一根两端同时烧（30分钟），同时第二根一端烧。第一根烧完后点燃第二根另一端（15分钟）。共45分钟。"},
]

ALL_QUESTIONS = MATH_QUESTIONS + LOGIC_QUESTIONS


def print_benchmark_summary():
    """打印benchmark概览"""
    print("=" * 70)
    print("PSOA Benchmark - 推理能力测试集")
    print("=" * 70)
    print()
    print("总题数: " + str(len(ALL_QUESTIONS)))
    print("  数学题: " + str(len(MATH_QUESTIONS)) + " 道 (GSM8K风格)")
    print("  逻辑题: " + str(len(LOGIC_QUESTIONS)) + " 道")
    print()
    print("难度分布:")
    for diff in ["easy", "medium", "hard"]:
        count = sum(1 for q in ALL_QUESTIONS if q.get("difficulty") == diff)
        label = {"easy": "简单", "medium": "中等", "hard": "困难"}[diff]
        print("  " + label + ": " + str(count))
    print()
    print("题目列表:")
    for q in ALL_QUESTIONS:
        qid = q["id"]
        diff = q["difficulty"]
        trap = " [陷阱题]" if "trap" in q else ""
        qtext = q["question"][:50]
        print("  [" + qid + "] (" + diff + ") " + qtext + "..." + trap)
    print()
    print("=" * 70)
    print("使用说明:")
    print("  1. 将每题作为PSOA架构的问题输入")
    print("  2. 记录收敛情况、轮数、最终答案")
    print("  3. 将模型答案与expected字段对比")
    print("  4. 统计正确率、平均轮数、致命错误率等指标")
    print("=" * 70)


def export_questions_json(output_path="benchmark_questions.json"):
    """导出为JSON格式"""
    data = {
        "math_questions": MATH_QUESTIONS,
        "logic_questions": LOGIC_QUESTIONS,
        "all_questions": ALL_QUESTIONS,
        "metadata": {
            "total": len(ALL_QUESTIONS),
            "math_count": len(MATH_QUESTIONS),
            "logic_count": len(LOGIC_QUESTIONS),
            "created": "2026-07-01",
            "purpose": "PSOA v2 benchmark for reasoning evaluation"
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Questions exported to: " + output_path)


if __name__ == "__main__":
    print_benchmark_summary()
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        export_questions_json()
