import torch.nn as nn
import torch
import torch.nn.functional as F



class LayoutfLowEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.hidden_dim = kwargs.get("hidden_size", 768)
        
        # 文字部分的 token id 嵌入层
        self.text_ids_embedding = nn.Embedding(kwargs.get("vocab_size", 30522), self.hidden_dim)
        
        # 文字部分 token 一维嵌入层
        self.sequence_len = kwargs.get("max_sequence_length", 512)
        
        text_position = torch.range(0, self.sequence_len, dtype=torch.long).view(1, -1)
        self.register_buffer("text_position", text_position)
        
        self.padding_idx = kwargs.get("padding_idx", 0)
        self.text_position_embedding = nn.Embedding(self.sequence_len+1, kwargs.get("hidden_size", 768), padding_idx=self.padding_idx)
        
        # bbox 部分的嵌入层
        
        self.max_2d_positon_embedding = kwargs.get("max_2d_position", 1024) # bbox所在图像上的最大尺寸为 1024x1024
        self.pos2d_dim = self.hidden_dim// 6 # bbox zuo
        
        self.x_positon_embedding = nn.Embedding(self.max_2d_positon_embedding, self.pos2d_dim)
        self.y_positon_embedding = nn.Embedding(self.max_2d_positon_embedding, self.pos2d_dim)
        self.h_positon_embedding = nn.Embedding(self.max_2d_positon_embedding, self.pos2d_dim)
        self.w_positon_embedding = nn.Embedding(self.max_2d_positon_embedding, self.pos2d_dim)


        # 图像部分的嵌入层
        self.patch_size = kwargs.get("patch_size", 16)
        self.img_projection = nn.Conv2d(3, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, input_ids,bbox, img):
        """
        这里默认 img.shape = [batch_size, 3, img_size=224, img_size=224]
        """
        img = self.img_projection(img) # [batch_size, hidden_dim, img_size/patch_size, img_size/patch_size]
        img_embedding = img.flatten(2).permute(0, 2, 1) # img.shape = [batch_size, num_patches, hidden_dim]
        

        # token 嵌入
        # =========> input_ids_embedding.shape  = [batch_size, sequence_length, hidden_dim]
        input_ids_embedding = self.text_ids_embedding(input_ids) # [batch_size, sequence_length, hidden_dim]
        
        # token 位置 嵌入
        input_mask = input_ids.ne(self.padding_idx).int()
        input_ids_pos = torch.cumsum(input_mask, dim=1) * input_mask
        # =========> input_ids_pos_embedding.shape  = [batch_size , sequence_length, hidden_dim]
        input_ids_pos_embedding = self.text_position_embedding(input_ids_pos)
        
        # bbox 嵌入
        bbox_left = bbox[:, :, 0]
        bbox_top = bbox[:, :, 1]
        bbox_right = bbox[:, :, 2]
        bbox_bottom = bbox[:, :, 3]
        bbox_height = torch.clip(bbox_bottom - bbox_top, 1, self.max_2d_positon_embedding-1)
        bbox_width = torch.clip(bbox_right - bbox_left, 1, self.max_2d_positon_embedding-1)
        
        bbox_left_embedding = self.x_positon_embedding(bbox_left)# [batch_size, sequence_length, pos2d_dim]
        bbox_top_embedding = self.y_positon_embedding(bbox_top)
        bbox_right_embedding = self.x_positon_embedding(bbox_right)
        bbox_bottom_embedding = self.y_positon_embedding(bbox_bottom)
        bbox_height_embedding = self.h_positon_embedding(bbox_height)
        bbox_width_embedding = self.w_positon_embedding(bbox_width)
        
        # ===========> pos2d_embeding.shape = [batch_size, sequence_length, pos2d_dim*6=hidden_dim]
        pos2d_embeding = torch.cat([bbox_left_embedding, bbox_top_embedding, bbox_right_embedding, bbox_bottom_embedding,bbox_height_embedding, bbox_width_embedding], dim=-1) 
        
        # 最终的嵌入结果
        # ===========> embedding.shape = [batch_size, sequence_length, hidden_dim]
        embedding = input_ids_embedding + input_ids_pos_embedding + pos2d_embeding 
        
        text_bbox_img = torch.cat([embedding, img_embedding], dim=1) # [batch_size, sequence_length + num_patches, hidden_dim]

        return text_bbox_img


















class LayoutFlow(nn.Module):    
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = LayoutfLowEmbedding()
        self.model =nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=kwargs.get("hidden_size", 768), nhead=kwargs.get("num_attention_heads", 12)),
            num_layers=kwargs.get("num_hidden_layers", 12)
        )
  
        
        
    def forward(self, x):
        # 送入模型前待处理的数据
        text_ids = x["input_ids"]
        text_mask = x["attention_mask"]
        text_bbox = x["bbox"]
        image = x["image"] # [batch_size, 3, img_size=224, img_size=224]
        labels = x["labels"]
        # 数据处理（成为可以送入模型的数据格式）
        mix_img = self.embedding(text_ids, text_bbox, image)
        out = self.model(mix_img)
        print(out.shape)
        
if __name__ == "__main__":
    model = LayoutFlow()
    x = {
        "input_ids": torch.randint(0, 30522, (2, 512)),
        "attention_mask": torch.ones((2, 512)),
        "bbox": torch.randint(0, 1024, (2, 512, 4)),
        "image": torch.randn((2, 3, 224, 224)),
        "labels": torch.randint(0, 2, (2, 512))
    }
    model(x)
        
        
        
        
        
        
        
        
        
        
        






















