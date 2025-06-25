import torch
class Mediate():
    def __init__(self,
                 ori_m1,
                 ori_m2):
        self.ori_m1=ori_m1
        self.ori_m2=ori_m2

    def do_m(self):
        max_m1=self.ori_m1.abs().amax(dim=(0,2,3),keepdim=True)
        max_m2=self.ori_m2.abs().amax(dim=(0,2,3),keepdim=True)
        med=torch.sqrt(max_m2/max_m1).abs()

        diag_matrix = med
        
        diag_inv = 1.0 / med
        
        m1_adjusted = self.ori_m1 * diag_matrix
        m2_adjusted = self.ori_m2 * diag_inv
        
        return m1_adjusted, m2_adjusted, diag_matrix, diag_inv
    
if __name__ == "__main__":
    # 生成具有不同range的张量来模拟真实情况
    # m1的range为1，m2的range为20
    m1 = torch.randn(100, 20, 256, 64) * 0.5  # 范围约为[-1, 1]
    m2 = torch.randn(100, 20, 256, 64) * 10   # 范围约为[-20, 20]
    
    med = Mediate(m1, m2)
    m1_adj, m2_adj, diag, diag_inv = med.do_m()
    
    # 验证调和效果
    print("\n=== 验证调和效果 ===")
    print(f"m1原始最大值: {m1.abs().amax():.4f}")
    print(f"m2原始最大值: {m2.abs().amax():.4f}")
    print(f"m1调整后最大值: {m1_adj.abs().amax():.4f}")
    print(f"m2调整后最大值: {m2_adj.abs().amax():.4f}")
    # 验证乘积是否相等
    ori_product = m1 * m2
    adj_product = m1_adj * m2_adj
    diff = ori_product - adj_product
    print(f"\n=== 乘积验证 ===")
    print(f"乘积差异绝对值和: {diff.abs().sum():.6f}")
    print(f"乘积是否相等: {torch.allclose(ori_product, adj_product, atol=1e-6)}")