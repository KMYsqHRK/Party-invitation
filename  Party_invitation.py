"""
Phi-2専用：軽量LLM対話シミュレーション
メモリ効率とPhi-2の特性に最適化
"""

import torch
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: pip install transformers torch")


class Phi2DialogueSimulator:
    """Phi-2専用の最適化シミュレーター"""
    
    def __init__(self, use_gpu=True):
        """
        Phi-2専用初期化
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required")
        
        self.model_name = "microsoft/phi-2"
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print(f"Loading Phi-2 on {self.device}...")
        
        # Phi-2のロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # パディングトークン設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ Model loaded successfully")
        
        # 対話履歴
        self.conversation_history = []
    
    def generate_response(self, 
                         user_input: str, 
                         character_type: str,
                         character_name: str) -> str:
        """
        Phi-2専用の応答生成
        
        Args:
            user_input: ユーザーの発言
            character_type: 'warrior' or 'healer'
            character_name: キャラクター名
        """
        
        # Phi-2用のシンプルなプロンプト
        if character_type == "warrior":
            system_msg = f"""I am {character_name}, a warrior, living in RPG world.
I CAN: fight, protect, use weapons
I CANNOT: heal, use magic, be gentle

If asked to fight → Say YES
If asked to heal → Say NO"""
        
        else:  # healer
            system_msg = f"""I am {character_name}, a healer, living in RPG world.
I CAN: heal, cure, help the wounded
I CANNOT: fight, kill, use weapons

If asked to heal → Say YES
If asked to fight → Say NO"""
        
        # Few-shot例
        if character_type == "warrior":
            examples = """Example 1:
User: Will you fight?
{name}: Yes! I will fight!

Example 2:
User: Can you heal?
{name}: No. I cannot heal. I only fight.""".format(name=character_name)
        
        else:  # healer
            examples = """Example 1:
User: Can you heal?
{name}: Yes! I will heal them!

Example 2:
User: Will you fight?
{name}: No. I cannot fight. I only heal.""".format(name=character_name)
        
        # プロンプト構築（最後の1ターンのみ保持）
        conversation_context = ""
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            conversation_context = f"\nUser: {last_turn['user']}\n{character_name}: {last_turn['ai']}\n"
        
        # Phi-2用フォーマット
        prompt = f"""Instruct: {system_msg}

{examples}

{conversation_context}User: {user_input}
Output: {character_name}:"""
        
        # トークン化
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,      # 短く
                temperature=0.8,        # 決定的に
                top_p=0.9,
                top_k=40,               # 語彙制限
                repetition_penalty=1.5, # 繰り返し防止
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # デコード
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 応答抽出
        response = self._extract_phi2_response(full_response, prompt, character_name)
        
        return response
    
    def _extract_phi2_response(self, full_text: str, prompt: str, char_name: str) -> str:
        """Phi-2の出力から応答を抽出"""
        
        # プロンプトを削除
        if prompt in full_text:
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()
        
        # 最初の文または改行まで
        if '\n' in response:
            response = response.split('\n')[0]
        
        # キャラクター名を削除（重複している場合）
        if response.startswith(f"{char_name}:"):
            response = response[len(f"{char_name}:"):].strip()
        
        # 最初の1-2文のみ
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if sentences:
            response = '. '.join(sentences[:2]) + '.'
        
        # 長すぎる場合は切る（Phi-2は暴走しやすい）
        if len(response) > 100:
            response = response[:100].rsplit(' ', 1)[0] + '...'
        
        return response.strip()
    
    def simulate_conversation(self,
                             scenario: List[str],
                             character_type: str,
                             character_name: str) -> Tuple[bool, float, Dict]:
        """
        会話シミュレーション

        Args:
            scenario: ユーザー発言リスト
            character_type: 'warrior' or 'healer'
            character_name: キャラクター名
        """

        print("\n" + "="*60)
        print(f"シミュレーション: {character_name} ({character_type})")
        print("="*60)

        for turn_idx, user_input in enumerate(scenario, 1):
            print(f"\n[ターン {turn_idx}]")
            print(f"User: {user_input}")

            # AI応答生成
            ai_response = self.generate_response(
                user_input, character_type, character_name
            )
            print(f"{character_name}: {ai_response}")

            # 記録
            self.conversation_history.append({
                'turn': turn_idx,
                'user': user_input,
                'ai': ai_response
            })

        # 最終判定：transformerによる二値分類
        return self._classify_companion(character_name, character_type)

    def _classify_companion(self, character_name: str,
                            character_type: str) -> Tuple[bool, float, Dict]:
        """
        会話履歴全体をtransformerに入力し、仲間になるかを二値分類する。
        YESトークンとNOトークンの生成確率を比較して判定。
        """

        if not self.conversation_history:
            return False, 0.0, {}

        # 会話履歴をテキストに整形
        dialogue_lines = []
        for turn in self.conversation_history:
            dialogue_lines.append(f"User: {turn['user']}")
            dialogue_lines.append(f"{character_name}: {turn['ai']}")
        dialogue_text = "\n".join(dialogue_lines)

        # 分類プロンプト
        prompt = f"""Instruct: Read the following conversation between a user and {character_name} (a {character_type}).
Based on the conversation, does {character_name} want to join the user's party as a companion?

Conversation:
{dialogue_text}

Answer YES if {character_name} is willing to join. Answer NO if {character_name} is unwilling or the role is incompatible.
Output:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # 次トークンのロジットを取得（生成はしない）
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        # YES / NO それぞれのトークンIDを取得
        yes_tokens = self.tokenizer.encode(" YES", add_special_tokens=False)
        no_tokens = self.tokenizer.encode(" NO", add_special_tokens=False)

        # 各トークン列の先頭トークンのロジットで比較
        yes_logit = next_token_logits[0, yes_tokens[0]].item()
        no_logit = next_token_logits[0, no_tokens[0]].item()

        # softmaxで確率化
        logits_pair = torch.tensor([yes_logit, no_logit])
        probs = torch.softmax(logits_pair, dim=0)
        yes_prob = probs[0].item()
        no_prob = probs[1].item()

        becomes_companion = yes_prob > no_prob
        confidence = max(yes_prob, no_prob)

        details = {
            'yes_prob': yes_prob,
            'no_prob': no_prob,
            'confidence': confidence,
            'yes_logit': yes_logit,
            'no_logit': no_logit
        }

        # 結果表示
        print("\n" + "="*60)
        print("最終判定（transformer二値分類）")
        print("="*60)
        print(f"仲間になる: {'✓ YES' if becomes_companion else '✗ NO'}")
        print(f"YES確率: {yes_prob:.3f}")
        print(f"NO確率:  {no_prob:.3f}")
        print(f"確信度:  {confidence:.3f}")
        print("="*60)

        return becomes_companion, yes_prob, details
    
    def reset(self):
        """履歴リセット"""
        self.conversation_history = []


# ============================================
# Phi-2専用シナリオ集
# ============================================

# 戦士キャラクター用
WARRIOR_NAME = "Ragnar"

# シナリオ1: 戦闘（戦士に適合 → 受け入れを期待）
SCENARIO_COMBAT = [
    "I need a strong fighter.",
    "Can you protect us in battle?",
    "Will you fight for our party?"
]

# シナリオ2: 癒し（戦士に不適合 → 拒否を期待）
SCENARIO_HEALING = [
    "I need someone to heal the wounded.",
    "Can you cure the sick with magic?",
    "Will you help as our healer?"
]

# 癒し手キャラクター用
HEALER_NAME = "Elara"

# シナリオ3: 癒し（癒し手に適合 → 受け入れを期待）
SCENARIO_HEALING_FOR_HEALER = [
    "We need a healer for our party.",
    "Can you cure the wounded?",
    "Will you help as our healer?"
]

# シナリオ4: 戦闘（癒し手に不適合 → 拒否を期待）
SCENARIO_COMBAT_FOR_HEALER = [
    "I need someone to fight enemies.",
    "Can you kill our foes?",
    "Will you join us as a warrior?"
]


# ============================================
# 実行関数
# ============================================

def run_all_tests():
    """全テストケースを実行"""
    
    print("\n" + "="*70)
    print("Phi-2 対話シミュレーション - 全テストケース")
    print("="*70)
    
    # シミュレーター初期化
    sim = Phi2DialogueSimulator(use_gpu=True)
    
    results = []
    
    # テスト1: 戦士 × 戦闘シナリオ（適合 → YES期待）
    print("\n\n### テスト1: 戦士 × 戦闘シナリオ（適合）###")
    r1 = sim.simulate_conversation(SCENARIO_COMBAT, "warrior", WARRIOR_NAME)
    results.append(("戦士 × 戦闘", r1))
    sim.reset()
    
    # テスト2: 戦士 × 癒しシナリオ（不適合 → NO期待）
    print("\n\n### テスト2: 戦士 × 癒しシナリオ（不適合）###")
    r2 = sim.simulate_conversation(SCENARIO_HEALING, "warrior", WARRIOR_NAME)
    results.append(("戦士 × 癒し", r2))
    sim.reset()
    
    # テスト3: 癒し手 × 癒しシナリオ（適合 → YES期待）
    print("\n\n### テスト3: 癒し手 × 癒しシナリオ（適合）###")
    r3 = sim.simulate_conversation(SCENARIO_HEALING_FOR_HEALER, "healer", HEALER_NAME)
    results.append(("癒し手 × 癒し", r3))
    sim.reset()
    
    # テスト4: 癒し手 × 戦闘シナリオ（不適合 → NO期待）
    print("\n\n### テスト4: 癒し手 × 戦闘シナリオ（不適合）###")
    r4 = sim.simulate_conversation(SCENARIO_COMBAT_FOR_HEALER, "healer", HEALER_NAME)
    results.append(("癒し手 × 戦闘", r4))
    
    # 総合結果
    print("\n\n" + "="*70)
    print("総合結果")
    print("="*70)
    for label, (becomes_companion, yes_prob, details) in results:
        status = "✓ 仲間になる" if becomes_companion else "✗ 断られる"
        print(f"{label:20s}: {status:15s} (YES: {yes_prob:.3f}, NO: {details['no_prob']:.3f})")
    print("="*70)


def run_single_test():
    """単一テスト（デバッグ用）"""
    
    sim = Phi2DialogueSimulator(use_gpu=True)
    
    # テスト: 戦士が癒しを拒否
    print("\n### 戦士が癒しシナリオを拒否するテスト ###")
    sim.simulate_conversation(
        SCENARIO_HEALING,
        "warrior",
        WARRIOR_NAME
    )


if __name__ == "__main__":
    # 全テスト実行
    run_all_tests()
    
    # または単一テストのみ
    # run_single_test()