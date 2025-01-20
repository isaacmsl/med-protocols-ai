from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()

examples = [
    (
        "tabagismo precisa de exames para diagnóstico?",
        "O diagnóstico do tabagismo é clínico e não necessita de exames, sendo feito por meio de avaliação do profissional de saúde com base no relato do paciente.",
    ),
    (
        "dm2 requer exames para diagnóstico?",
        "Sim, o diagnóstico de diabetes mellitus tipo 2 (DM2) requer exames complementares para confirmação, além da avaliação clínica. O rastreamento da DM2 também envolve exames laboratoriais.",
    ),
    (
        "o que é hemoglobina glicada?",
        "A hemoglobina glicada (HbA1c) é um exame de sangue que mede a quantidade de glicose ligada à hemoglobina. É um importante marcador para o controle do diabetes mellitus.",
    ),
    (
        "epilepsia precisa de exames complementares?",
        "O diagnóstico de epilepsia é primariamente clínico, baseado na descrição da crise epiléptica. Ademais, o diagnóstico de epilepsia pode necessitar de exames complementares, além da avaliação clínica. O principal exame complementar é o eletroencefalograma (EEG), que auxilia o médico a estabelecer um diagnóstico mais preciso.",
    ),
    (
        "qual remédio para idoso diabético é recomendado?",
        "Para o tratamento de diabetes mellitus tipo 2 (DM2) em idosos, a metformina é geralmente considerada o medicamento de primeira escolha devido à sua eficácia, segurança e baixo risco de hipoglicemia, além de trazer benefícios na redução do risco cardiovascular.",
    ),
    (
        "qual é a dose de carbamazepina para criança de 5 anos?",
        "Para crianças com menos de 6 anos de idade, a dose inicial de carbamazepina é de 5 a 10 mg/kg/dia. O aumento da dose deve ser feito gradualmente, cerca de 5-10 mg/kg/dia por semana. A dose máxima recomendada para crianças nessa faixa etária é de 35 mg/kg/dia.",
    ),
    (
        "qual é dose máxima diária do Cloridrato de Bupropiona?",
        "A dose máxima diária recomendada de cloridrato de bupropiona é de 300 mg, dividida em duas tomadas de 150 mg.",
    ),
]

inputs = [{"question": input_prompt} for input_prompt, _ in examples]
outputs = [{"answer": output_answer} for _, output_answer in examples]

dataset = client.create_dataset(
    dataset_name="PCDT Perguntas e Respostas",
    description="Um dataset com perguntas e respostas sobre os PCDTs.",
)

client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
