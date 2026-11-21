// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title LiquidGovernance
 * @notice Mo hinh quan tri DAO uy quyen long (Liquid Democracy) kem kiem tra chu trinh
 * @dev Dung de do chi phi Gas thuc te cho Muc 5 BTL Blockchain
 */
contract LiquidGovernance {
    struct Proposal {
        string description;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
    }

    address public admin;
    uint256 public maxDelegationDepth = 5; // Gioi han do sau uy quyen de tranh vuot Block Gas Limit

    mapping(address => uint256) public tokenWeight;
    mapping(address => address) public delegateTo;
    mapping(address => address[]) public delegatorsOf;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    Proposal[] public proposals;
    address[] public registeredVoters;

    event VoterRegistered(address indexed voter, uint256 weight);
    event Delegated(address indexed from, address indexed to);
    event DelegationRevoked(address indexed voter);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support, uint256 effectivePower);

    constructor() {
        admin = msg.sender;
    }

    // 1. Dang ky so du token bieu quyet cho cu tri (mo phong snapshot)
    function registerVoter(address _voter, uint256 _weight) external {
        require(msg.sender == admin, "Chi admin duoc cap quyen");
        if (tokenWeight[_voter] == 0 && _weight > 0) {
            registeredVoters.push(_voter);
        }
        tokenWeight[_voter] = _weight;
        emit VoterRegistered(_voter, _weight);
    }

    // 2. Uy quyen long bac cau (co kiem tra chong vong lap vo han A -> B -> A)
    function delegate(address _to) external {
        require(tokenWeight[msg.sender] > 0, "Khong co token bieu quyet");
        require(_to != msg.sender, "Khong the tu uy quyen cho chinh minh");
        require(_to != address(0), "Dia chi khong hop le");

        // Duyet chuoi uy quyen de phat hien chu trinh (Cycle Detection)
        address current = _to;
        uint256 depth = 0;
        while (delegateTo[current] != address(0)) {
            current = delegateTo[current];
            require(current != msg.sender, "Loi: Phat hien chu trinh uy quyen!");
            depth++;
            require(depth <= maxDelegationDepth, "Vuot qua do sau uy quyen cho phep");
        }

        // Xoa uy quyen cu neu co
        if (delegateTo[msg.sender] != address(0)) {
            _removeDelegator(delegateTo[msg.sender], msg.sender);
        }

        delegateTo[msg.sender] = _to;
        delegatorsOf[_to].push(msg.sender);
        emit Delegated(msg.sender, _to);
    }

    // 3. Thu hoi uy quyen bat ky luc nao
    function revokeDelegation() external {
        address currentDelegate = delegateTo[msg.sender];
        require(currentDelegate != address(0), "Ban chua uy quyen cho ai");
        _removeDelegator(currentDelegate, msg.sender);
        delegateTo[msg.sender] = address(0);
        emit DelegationRevoked(msg.sender);
    }

    // 4. Tinh tong quyen bieu quyet hieu dung (de quy theo cay uy quyen)
    function getEffectiveVotingPower(address _voter) public view returns (uint256) {
        // Neu da uy quyen cho nguoi khac thi quyen bieu quyet truc tiep bang 0
        if (delegateTo[_voter] != address(0)) {
            return 0;
        }
        return _sumSubtreePower(_voter, 0);
    }

    function _sumSubtreePower(address _node, uint256 _depth) internal view returns (uint256) {
        require(_depth <= maxDelegationDepth, "Vuot do sau cay uy quyen");
        uint256 total = tokenWeight[_node];
        address[] memory subs = delegatorsOf[_node];
        for (uint256 i = 0; i < subs.length; i++) {
            total += _sumSubtreePower(subs[i], _depth + 1);
        }
        return total;
    }

    // 5. Tao de xuat va bo phieu On-chain
    function createProposal(string calldata _description) external {
        proposals.push(Proposal({
            description: _description,
            yesVotes: 0,
            noVotes: 0,
            executed: false
        }));
    }

    function vote(uint256 _proposalId, bool _support) external {
        require(_proposalId < proposals.length, "De xuat khong ton tai");
        require(!hasVoted[_proposalId][msg.sender], "Ban da bo phieu roi");
        require(delegateTo[msg.sender] == address(0), "Ban da uy quyen, hay thu hoi truoc khi vote");

        uint256 power = getEffectiveVotingPower(msg.sender);
        require(power > 0, "Quyen bieu quyet bang 0");

        hasVoted[_proposalId][msg.sender] = true;
        if (_support) {
            proposals[_proposalId].yesVotes += power;
        } else {
            proposals[_proposalId].noVotes += power;
        }
        emit Voted(_proposalId, msg.sender, _support, power);
    }

    function _removeDelegator(address _delegate, address _delegator) internal {
        address[] storage list = delegatorsOf[_delegate];
        for (uint256 i = 0; i < list.length; i++) {
            if (list[i] == _delegator) {
                list[i] = list[list.length - 1];
                list.pop();
                break;
            }
        }
    }
}