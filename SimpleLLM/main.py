import torch
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SimpleTextDataset(Dataset):
    def __init__(self, text, seq_length=10):
        self.token_size = 3
        self.text = text.lower()
        self.seq_length = seq_length

        padding_needed = (self.token_size - len(self.text) % self.token_size) % self.token_size
        self.padded_text = self.text + ' ' * padding_needed
        self.tokens = [self.padded_text[i:i+self.token_size] for i in range(0, len(self.padded_text), self.token_size)]

        unique_tokens = sorted(list(set(self.tokens)))
        self.token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
        self.idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens)
        
        # Vectorize the text and store vectors in self.data
        self.data = []
        for i in range(0, len(self.tokens) - seq_length):
            input_seq = self.tokens[i:i+seq_length]
            target_token = self.tokens[i+seq_length]

            input_indices = [self.token_to_idx[token] for token in input_seq]
            target_index = self.token_to_idx[target_token]
            
            self.data.append((input_indices, target_index))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_indices, target_index = self.data[idx]
        return torch.tensor(input_indices), torch.tensor(target_index)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def decode(self, indices):
        return ''.join([self.idx_to_token[idx.item()] for idx in indices])
    
    def encode(self, text):
        text = text.lower()
        tokens = [text[i:i+self.token_size] for i in range(0, len(text), self.token_size)]
        
        if len(tokens[-1]) < self.token_size:
            tokens[-1] = tokens[-1] + ' ' * (self.token_size - len(tokens[-1]))
        
        default_idx = 0
        indices = [self.token_to_idx.get(token, default_idx) for token in tokens]

        return indices

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output
    
    def generate_text(self, dataset, seed_text, length=100, temperature=1.0):
        self.eval()
    
        if len(seed_text) % dataset.token_size != 0:
            padding_needed = dataset.token_size - (len(seed_text) % dataset.token_size)
            seed_text = seed_text + ' ' * padding_needed
        
        seed_tokens = [seed_text[i:i+dataset.token_size] for i in range(0, len(seed_text), dataset.token_size)]
        
        if len(seed_tokens) < dataset.seq_length:
            padding_tokens = ['   '] * (dataset.seq_length - len(seed_tokens))
            seed_tokens = padding_tokens + seed_tokens
        
        if len(seed_tokens) > dataset.seq_length:
            seed_tokens = seed_tokens[-dataset.seq_length:]
        
        indices = torch.tensor([[dataset.token_to_idx.get(token, 0) for token in seed_tokens]])
        generated_text = seed_text
        
        num_tokens_to_generate = length // dataset.token_size
        for _ in range(num_tokens_to_generate):
            with torch.no_grad():
                output = self(indices)
                
                if temperature != 1.0:
                    output = output / temperature
                
                probabilities = torch.softmax(output, dim=1)
                
                next_token_idx = torch.multinomial(probabilities, 1)
                
                indices = torch.cat([indices[:, 1:], next_token_idx], dim=1)
                
                next_token = dataset.idx_to_token[next_token_idx.item()]
                generated_text += next_token
        
        return generated_text

def train_model(model, dataset, epochs=10, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample_text = model.generate_text(dataset, "Paese", length=50)
            print(f"Sample text: {sample_text}")
    
    return model

def run_example():
    sample_text = """
    L’Italia è il Paese che amo. Qui ho le mie radici, le mie speranze, i miei orizzonti. Qui ho imparato, da mio padre e dalla vita, il mio mestiere di imprenditore. Qui ho appreso la passione per la libertà.
    Ho scelto di scendere in campo e di occuparmi della cosa pubblica perché non voglio vivere in un Paese illiberale, governato da forze immature e da uomini legati a doppio filo a un passato politicamente ed economicamente fallimentare.
    Per poter compiere questa nuova scelta di vita, ho rassegnato oggi stesso le mie dimissioni da ogni carica sociale nel gruppo che ho fondato. Rinuncio dunque al mio ruolo di editore e di imprenditore per mettere la mia esperienza e tutto il mio impegno a disposizione di una battaglia in cui credo con assoluta convinzione e con la più grande fermezza.
    So quel che non voglio e, insieme con i molti italiani che mi hanno dato la loro fiducia in tutti questi anni, so anche quel che voglio. E ho anche la ragionevole speranza di riuscire a realizzarlo, in sincera e leale alleanza con tutte le forze liberali e democratiche che sentono il dovere civile di offrire al Paese una alternativa credibile al governo delle sinistre e dei comunisti.
    La vecchia classe politica italiana è stata travolta dai fatti e superata dai tempi. L’autoaffondamento dei vecchi governanti, schiacciati dal peso del debito pubblico e dal sistema di finanziamento illegale dei partiti, lascia il Paese impreparato e incerto nel momento difficile del rinnovamento e del passaggio a una nuova Repubblica. Mai come in questo momento l’Italia, che giustamente diffida di profeti e salvatori, ha bisogno di persone con la testa sulle spalle e di esperienza consolidata, creative ed innovative, capaci di darle una mano, di far funzionare lo Stato.
    Il movimento referendario ha condotto alla scelta popolare di un nuovo sistema di elezione del Parlamento. Ma affinché il nuovo sistema funzioni, è indispensabile che al cartello delle sinistre si opponga, un polo delle libertà che sia capace di attrarre a sé il meglio di un Paese pulito, ragionevole, moderno.
    Di questo polo delle libertà dovranno far parte tutte le forze che si richiamano ai principi fondamentali delle democrazie occidentali, a partire da quel mondo cattolico che ha generosamente contribuito all’ultimo cinquantennio della nostra storia unitaria. L’importante è saper proporre anche ai cittadini italiani gli stessi obiettivi e gli stessi valori che hanno fin qui consentito lo sviluppo delle libertà in tutte le grandi democrazie occidentali.
    Quegli obiettivi e quei valori che invece non hanno mai trovato piena cittadinanza in nessuno dei Paesi governati dai vecchi apparati comunisti, per quanto riverniciati e riciclati. Né si vede come a questa regola elementare potrebbe fare eccezione proprio l’Italia. Gli orfani i e i nostalgici del comunismo, infatti, non sono soltanto impreparati al governo del Paese. Portano con sé anche un retaggio ideologico che stride e fa a pugni con le esigenze di una amministrazione pubblica che voglia essere liberale in politica e liberista in economia.
    Le nostre sinistre pretendono di essere cambiate. Dicono di essere diventate liberaldemocratiche. Ma non è vero. I loro uomini sono sempre gli stessi, la loro mentalità, la loro cultura, i loro più profondi convincimenti, i loro comportamenti sono rimasti gli stessi. Non credono nel mercato, non credono nell’iniziativa privata, non credono nel profitto, non credono nell’individuo. Non credono che il mondo possa migliorare attraverso l’apporto libero di tante persone tutte diverse l’una dall’altra. Non sono cambiati. Ascoltateli parlare, guardate i loro telegiornali pagati dallo Stato, leggete la loro stampa. Non credono più in niente. Vorrebbero trasformare il Paese in una piazza urlante, che grida, che inveisce, che condanna.
    Per questo siamo costretti a contrapporci a loro. Perché noi crediamo nell’individuo, nella famiglia, nell’impresa, nella competizione, nello sviluppo, nell’efficienza, nel mercato libero e nella solidarietà, figlia della giustizia e della libertà.
    Se ho deciso di scendere in campo con un nuovo movimento, e se ora chiedo di scendere in campo anche a voi, a tutti voi - ora, subito, prima che sia troppo tardi - è perché sogno, a occhi bene aperti, una società libera, di donne e di uomini, dove non ci sia la paura, dove al posto dell’invidia sociale e dell’odio di classe stiano la generosità, la dedizione, la solidarietà, l’amore per il lavoro, la tolleranza e il rispetto per la vita.
    Il movimento politico che vi propongo si chiama, non a caso, Forza Italia. Ciò che vogliamo farne è una libera organizzazione di elettrici e di elettori di tipo totalmente nuovo: non l’ennesimo partito o l’ennesima fazione che nascono per dividere, ma una forza che nasce invece con l’obiettivo opposto; quello di unire, per dare finalmente all’Italia una maggioranza e un governo all’altezza delle esigenze più profondamente sentite dalla gente comune.
    Ciò che vogliamo offrire agli italiani è una forza politica fatta di uomini totalmente nuovi. Ciò che vogliamo offrire alla nazione è un programma di governo fatto solo di impegni concreti e comprensibili. Noi vogliamo rinnovare la società italiana, noi vogliamo dare sostegno e fiducia a chi crea occupazione e benessere, noi vogliamo accettare e vincere le grandi sfide produttive e tecnologiche dell’Europa e del mondo moderno. Noi vogliamo offrire spazio a chiunque ha voglia di fare e di costruire il proprio futuro, al Nord come al Sud vogliamo un governo e una maggioranza parlamentare che sappiano dare adeguata dignità al nucleo originario di ogni società, alla famiglia, che sappiano rispettare ogni fede e che suscitino ragionevoli speranze per chi è più debole, per chi cerca lavoro, per chi ha bisogno di cure, per chi, dopo una vita operosa, ha diritto di vivere in serenità. Un governo e una maggioranza che portino più attenzione e rispetto all’ambiente, che sappiano opporsi con la massima determinazione alla criminalità, alla corruzione, alla droga. Che sappiano garantire ai cittadini più sicurezza, più ordine e più efficienza.
    La storia d’Italia è ad una svolta. Da imprenditore, da cittadino e ora da cittadino che scende in campo, senza nessuna timidezza ma con la determinazione e la serenità che la vita mi ha insegnato, vi dico che è possibile farla finita con una politica di chiacchiere incomprensibili, di stupide baruffe e di politica senza mestiere. Vi dico che è possibile realizzare insieme un grande sogno: quello di un’Italia più giusta, più generosa verso chi ha bisogno più prospera e serena più moderna ed efficiente protagonista in Europa e nel mondo.
    Vi dico che possiamo, vi dico che dobbiamo costruire insieme per noi e per i nostri figli, un nuovo miracolo italiano.
    """
    sample_text = re.sub(r'\s+', ' ', sample_text).strip()

    dataset = SimpleTextDataset(sample_text, seq_length=10)
    
    model = SimpleLLM(vocab_size=dataset.vocab_size, embedding_dim=48, hidden_dim=96)
    
    trained_model = train_model(model, dataset, epochs=500, batch_size=16, lr=0.0001)
    
    generated_text = trained_model.generate_text(dataset, "Paese", length=150, temperature=0.8)
    print("\nOUTPUT:")
    print(generated_text)
    
    return trained_model, dataset

if __name__ == "__main__":
    model, dataset = run_example()